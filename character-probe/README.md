## Character Probe Experiment for Master Thesis "Improving Text Rendering in Image Generation"

This repository contains all the base code required to run the Character Probe Experiment.
The code has been adapted from [Kaushal & Mahowald](https://github.com/Ayushk4/character-probing-pytorch/tree/master) (Experiment 1 only).

The purpose of the Character Probe Experiment is to investigate the character information present in CLIP's token embeddings.
The results indicate that CLIP does indeed contain this information, even more so than some language models investigated by [Kaushal & Mahowald, 2022](https://arxiv.org/abs/2206.02608).  
However, this was not instructive for our main analysis since we use the final embeddings and not the token embeddings for the text conditioning process.

## Instructions

Experiments are run using this Colab Notebook:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/medha-hegde/master_thesis/blob/main/thesis_experiments.ipynb)

Make sure to connect to GPU!
This experiment takes about 4 minutes to complete. 

1. Run the first two cells to clone the repository and install all required dependencies. The notebook will automatically restart and look like this:

    ![step1_image](/preliminary%20experiments/readme_imgs/step1.png)
2. In the cell titled "Run Character Probe Experiment", choose the text model using the dropdown:
    
    ![step2_image](/character-probe/step2.png)
3. Run the cell. The experiment will be run from scratch and F1-scores are displayed for each letter.
4. Final aggregate results are printed in the last line.






