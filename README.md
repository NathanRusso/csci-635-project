# Trading Card Classifier

We constructed 3 machine learning models, a CNN, a ViT, and a MobileViT, to classify trading card images as one of 3 categories: Magic: The Gathering, Pokemon, or Yu-Gi-Oh.

Table of Contents:
* [Developers](#developers-group-7)
* [About Our Project](#about-our-project)
* [Image Datasets](#image-datasets)
* [Using the Application](#using-the-application)

## Developers (Group 7)

- Brady Friese (bmf4295)
- Nathan Russo (ndr8530)
- Tyler Alvarez (tma9531)

## About Our Project

This project is for CSCI-635 Introduction to Machine Learning.

We constructed three machine learning models to classify trading card images as one of three categories: 
- Magic: The Gathering
- Pokemon
- Yu-Gi-Oh

The three models that we built were:
- Convolutional Neural Network (CNN)
- Vision Transformer (ViT)
- Mobile Vision Transformer (MobileViT)

We obtained image card data from Hugging Face and Kaggle and stored the grayscale, normalized, flattened, and compressed image vectors in our dataset for model training and evaluation. We also developed a live image-tracking system to demonstrate our trading card classification models.

## Image Datasets

### Hugging Face Repository for All Trading Card Images

- All (https://huggingface.co/datasets/bmf4295/CSCI635_Final_Project)

### Original Sources of Trading Card Images

- Magic: The Gathering (https://huggingface.co/datasets/acidtib/tcg-mtg-cards)
- Pokemon (https://www.kaggle.com/datasets/ellimaaac/pokemon-tcg-all-image-cards)
- Yu-Gi-Oh (https://www.kaggle.com/datasets/ioexception/yugioh-cards)

## Using the Application

### Requirements

1. You **must** have **Python (3+)** installed. Downloads can be found at https://www.python.org/downloads/. To verify Python was installed, run...

```bash
python --version
```

2. You **should** use a Python **virtual environment**. To create and activate the environment, run...

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\activate
```

3. You **must** install all necessary **libraries**. To get the application requirements, run...

```bash
pip install -r requirements.txt
```

### Setup

TBD - Pulling the image data???

### Running

TMD - How we test out models???
