import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError

# Load environment variables from .env file
load_dotenv()

# Initialize the S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def generate_s3_object_key(pdf_filename: str, file_type: str, file_name: str) -> str:
    """
    Generate a structured S3 object key.

    Args:
        pdf_filename (str): Name of the PDF without extension (used as a folder).
        file_type (str): Type of file (e.g., 'markdown', 'images').
        file_name (str): The actual file name.

    Returns:
        str: A structured S3 object key.
    """
    return f"{pdf_filename}/{file_type}/{file_name}"  # Flat, easy-to-navigate structure

    
def upload_file_to_s3(file_path: str, source: str, metadata: dict = None) -> str:
    """
    Upload a file to S3 with structured naming.

    Args:
        file_path (str): Local path of the file to upload.
        source (str): The PDF filename (used as the main folder).
        metadata (dict): Optional metadata tags for the object.

    Returns:
        str: Public URL of the uploaded file.
    """

    # Extract the actual file name
    file_name = os.path.basename(file_path)

    # Extract the file extension
    file_extension = os.path.splitext(file_name)[1].lower()

    # Determine file type based on extension
    extension_to_type = {
        ".md": "markdown",
        ".txt": "text",
        ".png": "images",
        ".jpg": "images",
        ".jpeg": "images",
        ".pdf": "pdfs",
        ".html": "html"
    }
    
    # Infer file type automatically
    file_type = extension_to_type.get(file_extension, "other")

    # Generate a structured S3 object key
    object_key = f"{source}/{file_type}/{file_name}"

    try:
        # Upload file to S3
        s3_client.upload_file(
            file_path, S3_BUCKET_NAME, object_key,
            ExtraArgs={"Metadata": metadata or {}, "ServerSideEncryption": "AES256"}
        )
        return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{object_key}"
    except Exception as e:
        raise RuntimeError(f"Error uploading {file_path} to S3: {str(e)}")
    

def generate_presigned_url(object_key, expiration=3600):
    """
    Generate a presigned URL for an object in S3.

    Args:
        object_key (str): S3 object key (file path).
        expiration (int): Time in seconds before the link expires.

    Returns:
        str: Presigned URL to access the file.
    """
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": object_key},
            ExpiresIn=expiration,
        )
        return url
    except NoCredentialsError:
        print("Credentials not available.")
        return None
    
def fetch_markdown_from_s3(s3_path):
    """
    Reads a Markdown file from S3 into memory.
    
    Args:
        s3_path (str): S3 file path (e.g., "processed/2020Q4/markdown/2020Q4_10-K_with_images.md").
    
    Returns:
        str: Markdown content as a string.
    """
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_path)
        return obj["Body"].read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching {s3_path} from S3: {e}")
        return None