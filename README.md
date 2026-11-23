# Trading Card Classifier

We constructed 3 machine learning models, a CNN, a ViT, and a MobileViT, to classify trading card images as one of 3 categories: Magic: The Gathering, Pokemon, or Yu-Gi-Oh.

Table of Contents:
* [Developers](#developers-group-7)
* [About Our Project](#about-our-project)
* [Image Datasets and Models](#image-datasets-and-models)
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

## Image Datasets and Models

### Hugging Face Repository for All Trading Card Images and Models

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

All models are uploaded to the Hugging Face repository.
While the CNN and MobileViT are small enough for GitHub, the ViT is not.

1. You **must** download the **ViT** from Hugging Face. To get the model (+ the CNN and MobileViT), run...

```bash
python .\code\download_models.py
```

### Running

You can now classify trading card images using the models in two ways: the CLI and the live demo.

#### CLI Classifier

The CLI takes in a list of image paths and classifies each card using all 3 models.
There are 2 classifiers with their own set of models.
Neither classifier was better than the other on all fronts.
- Classifier #1 is the original, and its models were trained on perfect card scans/images.
- Classifier #2 is the newer one, and its models were trained on altered card images which were changed to simulate real life conditions.

1. To classify a list of images with the CLI #1 or #2, run either...

```bash
python .\code\classifier1.py --images image_1_path image_2_path ...
```
```bash
python .\code\classifier2.py --images image_1_path image_2_path ...
```

- Example #1:

```bash
python .\code\classifier1.py --images .\data\mtg\mtg_000001.jpg .\data\pokemon\pokemon_000001.jpg .\data\yugioh\yugioh_000001.jpg

[INFO] Importing libraries...
[INFO] Running application...
[INFO] Classifying image(s)...

Classification Image #1: .\data\mtg\mtg_000001.jpg
        CNN:          [99.991 0.008 0.001], Magic: The Gathering
        ViT:          [100.000 0.000 0.000], Magic: The Gathering
        MobileViT:    [99.897 0.000 0.103], Magic: The Gathering

Classification Image #2: .\data\pokemon\pokemon_000001.jpg
        CNN:          [0.298 99.687 0.014], Pokemon
        ViT:          [0.026 99.973 0.002], Pokemon
        MobileViT:    [0.000 99.977 0.023], Pokemon

Classification Image #3: .\data\yugioh\yugioh_000001.jpg
        CNN:          [0.130 0.001 99.869], Yu-Gi-Oh
        ViT:          [0.000 0.000 100.000], Yu-Gi-Oh
        MobileViT:    [0.008 0.000 99.992], Yu-Gi-Oh

```

- Example #2:

```bash
python .\code\classifier2.py --images .\data\mtg\mtg_000001.jpg .\data\pokemon\pokemon_000001.jpg .\data\yugioh\yugioh_000001.jpg

[INFO] Importing libraries...
[INFO] Running application...
[INFO] Classifying image(s)...

Classification Image #1: .\data\mtg\mtg_000001.jpg
        CNN:          [75.832 24.109 0.059], Magic: The Gathering
        ViT:          [5.511 94.489 0.000], Pokemon
        MobileViT:    [94.605 2.421 2.974], Magic: The Gathering

Classification Image #2: .\data\pokemon\pokemon_000001.jpg
        CNN:          [1.064 98.892 0.044], Pokemon
        ViT:          [3.031 96.931 0.038], Pokemon
        MobileViT:    [0.000 99.993 0.007], Pokemon

Classification Image #3: .\data\yugioh\yugioh_000001.jpg
        CNN:          [6.638 91.103 2.259], Pokemon
        ViT:          [3.525 96.474 0.000], Pokemon
        MobileViT:    [0.000 0.002 99.997], Yu-Gi-Oh

```

#### Live Demo

The live demo tracks card-shaped objects and draws a green box around them.
When you see a green box that lines up fairly well with the card, click the space bar to start the classification.
The demo will then display its predations on the screen.
It uses classifier #2.

2. To classify a live image captured from your camera, run...

```bash
python .\code\live_demo.py.
```
