from huggingface_hub import HfApi
from dotenv_vault import load_dotenv
import os


def push_to_huggingface():
    """
    Push the results folder to Hugging Face Hub.
    """
    # Ensure you have the correct token and repository ID
    # Replace 'TOKEN' with your actual Hugging Face token
    # Replace 'Player-Coach/YapaComparission3' with your actual repository ID
    load_dotenv()
    api = HfApi()
    api.upload_folder(
        folder_path="results",
        repo_id="Player-Coach/AccentClassification3",
        repo_type="model",
        token=os.getenv("HF_TOKEN"),
    )


if __name__ == "__main__":
    push_to_huggingface()
