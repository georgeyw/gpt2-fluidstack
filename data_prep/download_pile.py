import os

from constants import REPO_ID
from huggingface_hub import snapshot_download

DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(DIR, '../data/')

path = snapshot_download(repo_id=REPO_ID, 
                         repo_type="dataset", 
                         max_workers=16,
                         local_dir=LOCAL_DIR,
                         local_dir_use_symlinks=False,
                         cache_dir=LOCAL_DIR)

print(path)