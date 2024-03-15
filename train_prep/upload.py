import os
import asyncio
import aioboto3
from botocore.exceptions import NoCredentialsError

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")

S3_SESSION = aioboto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

async def upload_file_to_s3(s3_client, local_path, bucket_name, s3_path):
    try:
        await s3_client.upload_file(Filename=local_path, Bucket=bucket_name, Key=s3_path)
        print(f"Uploaded {local_path} to {s3_path}")
    except NoCredentialsError:
        print("Credentials not available")
        return
    
async def upload_folder_to_s3(local_directory, bucket_name, s3_folder):
    """
    Asynchronously uploads contents of a local directory to an S3 bucket path without deleting existing files,
    using the provided AWS credentials.
    
    Parameters:
    - local_directory: Path to the local directory to upload.
    - bucket_name: Name of the S3 bucket.
    - s3_folder: Path within the S3 bucket where files should be uploaded.
    - aws_access_key_id: AWS access key ID.
    - aws_secret_access_key: AWS secret access key.
    """
    
    async with S3_SESSION.client('s3') as s3_client:
        tasks = []
        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(s3_folder, relative_path)
                
                task = asyncio.ensure_future(upload_file_to_s3(s3_client, local_path, bucket_name, s3_path))
                tasks.append(task)
        
        await asyncio.gather(*tasks)
