# MetaCert: A Multimodal Classification Framework

## Table of Contents
- [Introduction](#introduction)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
  - [Environment Setup](#environment-setup)
  - [Dataset and Model Preparation](#dataset-and-model-preparation)
- [Usage](#usage)
- [Configuration](#configuration)

## Introduction

MetaCert (Metabolic Attention Network Utilizing Uncertainty Estimation) is an advanced framework designed to address the challenges in Multimodal Aspect-Category-Sentiment Triple Extraction (MACSTE), a complex subtask within Multimodal Aspect-Based Sentiment Analysis (MABSA). Part of this work and idea comes from dtca model. This framework focuses on simultaneous attribute extraction and sentiment polarity prediction from image-text pairs.

Key features of MetaCert include:

1. **Metabolic Attention Mechanism (MAM)**: Inspired by biochemical metabolic networks and enhanced with cross-attention, MAM optimizes information flow and exchange between modalities.

2. **Uncertainty Estimation Network (UEN)**: This component fine-tunes the semantic contributions of each modality while maintaining data integrity, thereby improving classification accuracy.

By addressing the often-overlooked aspects of information flow design and modality-specific noise, MetaCert achieves state-of-the-art (SOTA) results on the TWITTER-15 and TWITTER-17 datasets.

***This implementation has been developed as part of a research project submitted to the International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2025. We only upload the initial version for now, after we received the result of conference then will renew the full version.***


## Setup

### Environment Setup

1. Ensure you have Conda installed on your Windows 10 system.
2. Create a new Conda environment:
   ```
   conda create -n metacert python=3.9
   ```
3. Activate the environment:
   ```
   conda activate metacert
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Dataset and Model Preparation

1. Download the following files from [this Google Drive folder](https://drive.google.com/drive/folders/1ifXGGiuKIsf5ptLlGImTi9SQ-1kNHHCV?usp=drive_link):
   - `twitter1517_datasets.rar`
   - `models.rar`
   - `dualc.rar`

2. Extract the downloaded files and place their contents as follows:
   - Extract `twitter1517_datasets.rar` to the root of the `datasets` folder
   - Extract `models.rar` to the root of the `models` folder
   - Extract `dualc.rar` to `datasets/finetune` folder

## Folder Structure

After setting up the project, your folder structure should look like this:

```
metacert/
│
├── datasets/
│   ├── finetune/
│   │   └── dualc/
│   │       └── ... (dualc files)
│   ├── ITM/
│   ├── jsons/
│   ├── twitter2015/
│   ├── twitter2015_images/
│   ├── twitter2017/
│   └── twitter2017_images/
│
├── models/
│   ├── roberta-base/
│   ├── vit-base-patch16-224-in21k/
│   └── xlnet-base-cased/
│
├── caption.py
├── TrainInputProcessS.py
├── main.py
├── requirements.txt
└── README.md
```

## Usage

To run the program, use the following command:

```
python main.py --dataset_type 2017 --text_model_name roberta --image_model_name vit --batch_size 4 --alpha 0.3 --beta 0.0 --gamma 0.9
```

## Configuration

You can adjust the following parameters based on your requirements:

- `--dataset_type`: Specify the dataset you want to use
- `--text_model_name`: Choose the text model (e.g., roberta)
- `--image_model_name`: Choose the image model (e.g., vit)
- `--batch_size`: Set the batch size for training
- `--alpha`, `--beta`, `--gamma`: Adjust these hyperparameters as needed

Feel free to experiment with different combinations of datasets, pre-trained models, and hyperparameters to optimize the performance for your specific use case.
