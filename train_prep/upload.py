import os
import asyncio
import aioboto3
import threading

from botocore.exceptions import NoCredentialsError

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")

S3_SESSION = aioboto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

BUCKET = "devinterp-language"
S3_FOLDER = "checkpoints/gpt-2-small"

def start_asyncio_event_loop(loop):
    """Run an asyncio event loop in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def wait_for_remaining_tasks(loop):
    # Get all tasks in the current event loop
    tasks = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task(loop)]
    
    # If there are any tasks pending, wait for them to be completed.
    if tasks:
        await asyncio.gather(*tasks)

async def stop_loop(loop):
    await wait_for_remaining_tasks(loop)
    loop.stop()

def run_in_background(loop, coroutine):
    """Schedule coroutine execution in the asyncio event loop running in a separate thread."""
    asyncio.run_coroutine_threadsafe(coroutine, loop)
    asyncio.run_coroutine_threadsafe(stop_loop(loop), loop)

async def upload_file_to_s3(s3_client, local_path, bucket_name, s3_path):
    try:
        await s3_client.upload_file(Filename=local_path, Bucket=bucket_name, Key=s3_path)
        print(f"Uploaded {local_path} to {s3_path}")
    except NoCredentialsError:
        print("Credentials not available")
        return
    
async def upload_folder_to_s3(local_directory, bucket_name=BUCKET, s3_folder=S3_FOLDER):
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

def async_upload_to_s3(local_directory, bucket_name=BUCKET, s3_folder=S3_FOLDER):

    # Create a new asyncio event loop
    new_loop = asyncio.new_event_loop()

    # Start the new event loop in a background thread
    thread = threading.Thread(target=start_asyncio_event_loop, args=(new_loop,))
    thread.start()

    # Now you can run your async function in the background without blocking the main thread
    # Replace 'your_parameters_here' with the actual parameters your function needs
    coroutine = upload_folder_to_s3(local_directory, bucket_name, s3_folder)
    run_in_background(new_loop, coroutine)

    return thread
