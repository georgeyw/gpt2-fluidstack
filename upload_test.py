import os 

from train_prep.upload import async_upload_to_s3

folder1 = './upload_test/'
folder2 = './upload_test2/'

def create_test_files(num_files=2, size_in_gb=0.3, directory='./'):
    """
    Creates a set of test files of specified size.

    Parameters:
    - num_files: Number of files to create.
    - size_in_gb: Size of each file in gigabytes.
    - directory: The directory where files will be created.
    """
    size_in_bytes = int(size_in_gb * 1024 * 1024 * 1024)  # Convert GB to Bytes
    os.makedirs(directory, exist_ok=True)

    for i in range(num_files):
        filename = f"test_file_{i+1}.bin"
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            f.write(b'\0' * size_in_bytes)
        print(f"Created {filepath}")

# Example usage
create_test_files(directory=folder1 + 'test_folder1')
create_test_files(directory=folder1 + 'test_folder2')

create_test_files(directory=folder2 + 'test_folder3')
create_test_files(directory=folder2 + 'test_folder4')

thread1 = async_upload_to_s3(folder1)
thread2 = async_upload_to_s3(folder2)

print('Main thread still responsive!')
print('Foo bar baz bat qux quux corge grault garply waldo fred plugh xyzzy thud.')

thread1.join()
thread2.join()

print('Threads finished!')