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

    print(f"[INFO] Downloading Convolutional Neural Network (CNN) #2...")
    try:
        file_path = hf_hub_download(
            repo_id="bmf4295/CSCI635_Final_Project",
            repo_type="dataset",
            filename=cf.PATH_MODEL_CNN_2,
            local_dir="."
        )
        print(f"[INFO] Downloaded CNN #2 to: {file_path}\n")
    except Exception as e:
        print(f"[ERROR] Could not download CNN #2 to. The following error was given:\n{e}\n")
    
    print(f"[INFO] Downloading Vision Transformer (ViT) #2...")
    try:
        file_path = hf_hub_download(
            repo_id="bmf4295/CSCI635_Final_Project",
            repo_type="dataset",
            filename=cf.PATH_MODEL_VIT_2,
            local_dir="."
        )
        print(f"[INFO] Downloaded ViT #2 to: {file_path}\n")
    except Exception as e:
        print(f"[ERROR] Could not download ViT #2. The following error was given:\n{e}\n")

    print(f"[INFO] Downloading Mobile Vision Transformer (MobileViT) #2...")
    try:
        file_path = hf_hub_download(
            repo_id="bmf4295/CSCI635_Final_Project",
            repo_type="dataset",
            filename=cf.PATH_MODEL_MOBILEVIT_2,
            local_dir="."
        )
        print(f"[INFO] Downloaded MobileViT #2 to: {file_path}\n")
    except Exception as e:
        print(f"[ERROR] Could not download MobileViT #2. The following error was given:\n{e}\n")

    print(f"[INFO] Downloading Convolutional Neural Network (CNN) #3...")
    try:
        file_path = hf_hub_download(
            repo_id="bmf4295/CSCI635_Final_Project",
            repo_type="dataset",
            filename=cf.PATH_MODEL_CNN_3,
            local_dir="."
        )
        print(f"[INFO] Downloaded CNN #3 to: {file_path}\n")
    except Exception as e:
            print(f"[ERROR] Could not download CNN #3 to. The following error was given:\n{e}\n")

    print(f"[INFO] Downloading Vision Transformer (ViT) #3...")
    try:
        file_path = hf_hub_download(
            repo_id="bmf4295/CSCI635_Final_Project",
            repo_type="dataset",
            filename=cf.PATH_MODEL_VIT_3,
            local_dir="."
        )
        print(f"[INFO] Downloaded ViT #3 to: {file_path}\n")
    except Exception as e:
        print(f"[ERROR] Could not download ViT #3. The following error was given:\n{e}\n")

    print(f"[INFO] Downloading Mobile Vision Transformer (MobileViT) #3...")
    try:
        file_path = hf_hub_download(
            repo_id="bmf4295/CSCI635_Final_Project",
            repo_type="dataset",
            filename=cf.PATH_MODEL_MOBILEVIT_3,
            local_dir="."
        )
        print(f"[INFO] Downloaded MobileViT #3 to: {file_path}\n")
    except Exception as e:
        print(f"[ERROR] Could not download MobileViT #3. The following error was given:\n{e}\n")

if __name__ == "__main__":
    main()
