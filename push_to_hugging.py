from huggingface_hub import HfApi


def push_to_huggingface():
    """
    Push the results folder to Hugging Face Hub.
    """
    # Ensure you have the correct token and repository ID
    # Replace 'TOKEN' with your actual Hugging Face token
    # Replace 'Player-Coach/YapaComparission3' with your actual repository ID
    api = HfApi()
    api.upload_folder(
        folder_path="results",
        repo_id="Player-Coach/YapaComparission3",
        repo_type="model",
        token="TOKEN",
    )
