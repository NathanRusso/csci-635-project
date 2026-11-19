"""
Downloads the CNN, ViT, and Mobile ViT to the local project.
"""

from huggingface_hub import hf_hub_download
import config as cf

def main():
    """
    Pulls the CNN, ViT, and Mobile ViT models to your local computer from Hugging Face.
    """

    print(f"[INFO] Downloading Convolutional Neural Network (CNN)...")
    file_path = hf_hub_download(
        repo_id="bmf4295/CSCI635_Final_Project",
        repo_type="dataset",
        filename=cf.PATH_MODEL_CNN,
        local_dir="."
    )
    print(f"[INFO] Downloaded CNN to: {file_path}\n")
    
    print(f"[INFO] Downloading Vision Transformer (ViT)...")
    file_path = hf_hub_download(
        repo_id="bmf4295/CSCI635_Final_Project",
        repo_type="dataset",
        filename=cf.PATH_MODEL_VIT,
        local_dir="."
    )
    print(f"[INFO] Downloaded ViT to: {file_path}\n")
    
    print(f"[INFO] Downloading Mobile Vision Transformer (MobileViT)...")
    file_path = hf_hub_download(
        repo_id="bmf4295/CSCI635_Final_Project",
        repo_type="dataset",
        filename=cf.PATH_MODEL_MOBILEVIT,
        local_dir="."
    )
    print(f"[INFO] Downloaded MobileViT to: {file_path}\n")


if __name__ == "__main__":
    main()
