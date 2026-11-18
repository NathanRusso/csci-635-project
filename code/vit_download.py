"""
Downloads the Vision Transformer (ViT) to the local project since it is too large for Github.
"""

from huggingface_hub import hf_hub_download


def main():
    """
    Pulls the ViT model to your local computer from Hugging Face.
    """
    file_path = hf_hub_download(
        repo_id="bmf4295/CSCI635_Final_Project",
        repo_type="dataset",
        filename="data/635-ViT.keras",
        local_dir="."
    )
    print(f"Downloaded to: {file_path}")


if __name__ == "__main__":
    main()
