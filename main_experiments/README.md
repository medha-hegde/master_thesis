## Main Experiments for Master Thesis "Improving Text Rendering in Image Generation"

This repository contains all the base code required to run the Main Experiments.
The code has been adapted from [this](https://github.com/Animadversio/DiffusionFromScratch) repository.

The purpose of the Main Experiments is to observe the effect of certain architecturural aspect of the image geneartion model on its text rendering capabilities.
To do this, a diffusion model is trained from scratch by changing the following components:

1. Text Encoder Type: CLIP, T5 and ByT5
2. Text Encoder Size: CLIP Large, T5-base, ByT5-base
3. Training Time: Increased number of epochs
4. Combined Embeddings: All combinations of the smallest variants of CLIP, T5 and ByT5

The results indicate that large language models perform better at text rendering, particularity the character-level ByT5 model.
Increasing the size of these models also helps.
Increased training time for the CLIP-small model also improves results.
Combining embeddings with the ByT5 model's embeddings causes increase in OCR scores. 

## Instructions

Experiments are run using this Colab Notebook:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/medha-hegde/master_thesis/blob/main/thesis_experiments.ipynb)

Make sure to connect to GPU!

A [wandb](https://wandb.ai/home) account is required to run these experiments and record the training process.
Training time for each model is about 3 hours for 150 epochs. 

Steps:
1. Run the first two cells to clone the repository and install all required dependencies. The notebook will automatically restart and look like this:

    ![step1_image](/preliminary%20experiments/readme_imgs/step1.png)
2. In the cell titled "Main Experiments", choose the text model using the dropdown:
    
    ![step2_image](/main_experiments/readme_imgs/step2.png)

   The default number of epochs is 150. Increase this value for increased training time.
   
   The default additional text encoder is "None". Choose another text encoder here to run with combined embeddings. 
3. Run the cell. These are steps that occur:
   1. Training dataset of 7,000 image-text pairs are created.
   2. Diffusion model is trained on this dataset, and images are sampled and displayed after every epoch along with the loss value:
   
![training_image](/main_experiments/readme_imgs/training.png)

   3. After training is completed, the model is saved as **_ckpt_transformer_.pt** in this _main_experiments_ repository. 

The process is repeated for each of the models used in the Main Experiments. 





