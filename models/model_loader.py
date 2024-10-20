from huggingface_hub import snapshot_download, login
import yaml

my_token=""
repo_id = "casperhansen"
model_name = "llama-3-8b-instruct-awq"
save_dir = '/workspace/' + model_name

def get_token_from_config():
    with open('config.yaml') as f:
        file = yaml.safe_load(f)
        return file['hf_config']['token']

def download_model_to_folder():
    snapshot_download(
        repo_id=f"{repo_id}/{model_name}",
        local_dir=save_dir
    )

if __name__ == "__main__":
    my_token = get_token_from_config()
    login(my_token)
    download_model_to_folder()