from s3_utils import generate_presigned_url,s3_client,S3_BUCKET_NAME
from pinecone_rag import recursive_based_chunking,add_chunks_to_pinecone
from dotenv import load_dotenv
import os
from botocore.exceptions import NoCredentialsError
import requests

# Load environment variables
load_dotenv()


def get_all_markdown_files(bucket_name):
    """
    List all markdown files from all folders in the S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.

    Returns:
        list: List of markdown file keys.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            print("No files found in the bucket.")
            return []

        markdown_files = []
        for obj in response['Contents']:
            file_key = obj['Key']
            # Check if it's a markdown file inside a 'markdown/markdown/' folder structure
            if file_key.endswith('.md') and 'markdown/markdown/' in file_key:
                markdown_files.append(file_key)

        return markdown_files
    
    except NoCredentialsError:
        print("Credentials not available.")
        return []

def process_markdown_files(bucket_name):
    """
    Process all markdown files: generate presigned URLs, chunk content, and store in Pinecone.
    
    Args:
        bucket_name (str): Name of the S3 bucket.
    """
    # Get all markdown files from the bucket
    markdown_files = get_all_markdown_files(bucket_name)
    
    if not markdown_files:
        print("No markdown files found to process.")
        return
    
    for file_key in markdown_files:
        # Generate presigned URL for each file
        presigned_url = generate_presigned_url(file_key)
        
        if not presigned_url:
            print(f"Failed to generate presigned URL for file: {file_key}")
            continue
        
        print(f"Processing file: {file_key}")
        
        # Retrieve file content from presigned URL
        response = requests.get(presigned_url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve file from URL: {presigned_url}")
            continue
        
        document_text = response.text
        
        # Chunk content and add to Pinecone
        chunks = recursive_based_chunking(document_text)
        
        num_added = add_chunks_to_pinecone(chunks, presigned_url)
        
        print(f"Added {num_added} chunks from file: {file_key}")


if __name__ == "__main__":
    process_markdown_files(S3_BUCKET_NAME)