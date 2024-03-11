from constants import REPO_ID
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id=REPO_ID, repo_type="dataset", max_workers=16)

print(path)