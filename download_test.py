from huggingface_hub import snapshot_download

REPO_ID = "EleutherAI/pile-standard-pythia-preshuffled"
REPO_TYPE = "dataset"

path = snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE)

print(path)