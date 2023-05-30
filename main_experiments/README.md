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

Models are evaluated using OCR. 

## Instructions

Experiments are run using this Colab Notebook:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/medha-hegde/master_thesis/blob/main/thesis_experiments.ipynb)

Make sure to connect to GPU!

A [wandb](https://wandb.ai/home) account is required to run these experiments and record the training process.
Training time for each model is about 3 hours for 150 epochs. 

Steps:

Run the first two cells to clone the repository and install all required dependencies. The notebook will automatically restart and look like this:

    ![step1_image](/preliminary%20experiments/readme_imgs/step1.png)

### Model Training

1. In the cell titled "Main Experiments", choose the text model using the dropdown:
    
    ![step2_image](/main_experiments/readme_imgs/step2.png)

   The default number of epochs is 150. Increase this value for increased training time.
   
   The default additional text encoder is "None". Choose another text encoder here to run with combined embeddings. 
2. Run the cell. These are steps that occur:
   1. Training dataset of 7,000 image-text pairs are created.
   2. Diffusion model is trained on this dataset, and images are sampled and displayed after every epoch along with the loss value:
   
![training_image](/main_experiments/readme_imgs/training.png)

   3. After training is completed, the model is saved as **_ckpt_transformer_.pt** in this _main_experiments_ repository. 

The process is repeated for each of the models used in the Main Experiments. 

### OCR Evaluation

To evaluate the saved model, we generate 100 sample images using words from our test set and count the number of exact matches as evaluated by EasyOCR.   
Evaluation for one model takes about 1 hour.

1. In the cell titled "Run OCR ", input the required variables:
   1. model_name: Label you would like for your model
   2. model_saved_path = Path to the model .pt file
   3. text_encoder_name = Huggingface modelcard 
   4. combine_with_text_encoder = Huggingface modelcard for the 2nd text encoder, if running a combined embedding model
2. Run the cell. The 100 sample images for each word are displayed along with the OCR score:
![ocr_image](/main_experiments/readme_imgs/ocr.png)
3. After all 85 words, the results are saved in a .json file as _"ocr_results + model_name + .json"

Repeat for all required models. 

### OCR Plot

After generating the .json files for the 3 required models, the results can be displayed in an interactive plot. 

1. In the cell titled "Plot OCR Results", input the required variables:
   1. model_x_name: The model name to be displayed
   2. model_x_saved_path: The corresponding path the .json file generated in OCR Evaluation step above.
2. Run the cell. The output will look as follows:
![plot_ocr](/main_experiments/readme_imgs/ocr_plot.png)






