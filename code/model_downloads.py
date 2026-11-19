"""
Downloads the Vision Transformer (ViT) to the local project since it is too large for Github.
"""

from huggingface_hub import hf_hub_download
import config as cf

def main():
    """
    Pulls the ViT model to your local computer from Hugging Face.
    """
    
    print(f"Downloading Visual Transformer to: {file_path}")
    file_path = hf_hub_download(
        repo_id="bmf4295/CSCI635_Final_Project",
        repo_type="dataset",
        filename=cf.PATH_MODEL_VIT,
        local_dir="."
    )
    print(f"Downloaded to: {file_path}")
    
    print(f"Downloading Mobile Visual Transformer to: {file_path}")
    file_path = hf_hub_download(
        repo_id="bmf4295/CSCI635_Final_Project",
        repo_type="dataset",
        filename=cf.PATH_MODEL_MOBILEVIT,
        local_dir="."
    )
    print(f"Downloaded to: {file_path}")
    
    print(f"Downloading CNN to: {file_path}")
    file_path = hf_hub_download(
        repo_id="bmf4295/CSCI635_Final_Project",
        repo_type="dataset",
        filename=cf.PATH_MODEL_CNN,
        local_dir="."
    )
    print(f"Downloaded to: {file_path}")


if __name__ == "__main__":
    main()
