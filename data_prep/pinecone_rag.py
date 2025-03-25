import os
import re
import requests
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "nvidia-quarterly-reports"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load Sentence Transformer model (cached)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="huggingface_cache")

def embed_texts(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

def get_or_create_index():
    """
    Ensure that the specified Pinecone index exists and return the index object.
    """
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        print(f"Created new index: {INDEX_NAME}")
    return pc.Index(INDEX_NAME)

QUARTER_REGEX = re.compile(r"(Q[1-4])", re.IGNORECASE)
YEAR_REGEX = re.compile(r"(20\d{2})", re.IGNORECASE)

def extract_text(query: str, regex) -> str | None:
    match = regex.search(query)
    if match:
        result = match.groups()
        return f"{result[0]}"
    return None

def add_chunks_to_pinecone(chunks, markdown_file_path):
    """
    Add text chunks and their embeddings to the Pinecone index.
    """
    index = get_or_create_index()
 
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    ids = [f"{markdown_file_path}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": markdown_file_path, "chunk_index": i, "text": chunks[i], "quarter": extract_text(markdown_file_path,QUARTER_REGEX), "year": extract_text(markdown_file_path,YEAR_REGEX)} for i in range(len(chunks))]

    # Batch upsert the chunks
    total_chunks_added = 0
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        # Prepare vectors for upsert
        vectors = [
            (batch_ids[j], batch_embeddings[j].tolist(), batch_metadatas[j])
            for j in range(len(batch_ids))
        ]
        try:
            index.upsert(vectors=vectors)
            total_chunks_added += len(batch_ids)
        except Exception as e:
            print(f"Error upserting batch {i // batch_size + 1}: {e}")

    return total_chunks_added

def retrieve_relevant_chunks(query, metadata_filter=None, top_k=3, score_threshold=0.4):
    """
    Perform strict retrieval with metadata filtering.
    """
    index = get_or_create_index()
    query_embedding = embed_texts([query])[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter  
    )
    print(results)

    chunks, sources = [], []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            if "score" in match and match.get("score", 0) >= score_threshold:
                if "metadata" in match and "text" in match["metadata"]:
                    chunks.append(match["metadata"]["text"])
                    sources.append(match["metadata"]["source"])

    return chunks, sources

def recursive_based_chunking(text, max_chunk_size=300):
    """
    Split document text into chunks using RecursiveChunker.
    
    Args:
        text (str): The document text to chunk
        
    Returns:
        list: List of text chunks
    """
    # Initialize the chunker with our configured settings
    chunker = RecursiveTokenChunker(
        chunk_size=400,
        chunk_overlap=0,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    
    # Split the text into chunks
    chunks = chunker.split_text(text)
    
    # Return the chunks
    return chunks


def generate_response(query, chunks, sources):
    """
    Generate a response strictly from the retrieved document chunks.
    """
    openai.api_key=OPENAI_API_KEY
    
    if not chunks:
        return "No relevant information found."

    context = "\n\n".join([f"Source [{i+1}] ({source}): {chunk}" for i, (chunk, source) in enumerate(zip(chunks, sources))])

    system_message = "You are an assistant that only provides answers based on the provided document context."

    user_message = f"Question: {query}\n\nContext:\n{context}\n\nAnswer based strictly on the provided context."

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


def pinecone_rag_pipeline(s3_markdown_path, query,top_k):
    """
    Run the RAG pipeline while ensuring that answers are only retrieved from the uploaded document.
    """
    response = requests.get(s3_markdown_path)
    
    if response.status_code != 200:
        return "Failed to retrieve Markdown from S3."

    document_text = response.text

    chunks = recursive_based_chunking(document_text)

    add_chunks_to_pinecone(chunks, s3_markdown_path)

    
    retrieved_chunks, sources = retrieve_relevant_chunks(query, metadata_filter={"source": {"$eq": s3_markdown_path}}, top_k=top_k)

    if retrieved_chunks:
        response = generate_response(query, retrieved_chunks, sources)
        return response
    else:
        return "No relevant information found to answer the query."