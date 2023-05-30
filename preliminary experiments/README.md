# Preliminary Experiments for Master Thesis "Improving Text Rendering in Image Generation"
## June 2023, KU Leuven 
## Medha Hegde (r0872802)

This repository contains all the base code required to run the Preliminary Experiments.
The Preliminary Experiments are designed to investigate the nature of the CLIP text model's text encoding ability.
It reveals that while CLIP has OCR capabilities, it does not encode character level information.

1. Experiment 1: Increasing Font Size, Blank Background
2. Experiment 2: Increasing Font Size, Image Background
3. Experiment 3: Text Scrambling, In Image
4. Experiment 4: Text Scrambling, Text Only

## Instructions

Experiments are run using this Colab Notebook:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/medha-hegde/master_thesis/blob/main/thesis_experiments.ipynb)

Make sure to connect to GPU!
Each experiment takes 3-5 minutes to complete running. 


1. Run the first two cells to clone the repository and install all required dependencies. The notebook will automatically restart and look like this:
    ![step1_image](/preliminary%20experiments/readme_imgs/step1.png)
2. In the cell titled "Run Preliminary Experiment", choose the experiment using the dropdown.
3. Run the cell. The experiment will be run from scratch:
   1. Images (if required) are created and saved in a directory with the name of the experiment.
   2. CLIP Image/Text embeddings are calculated and saved in a directory named _'clip_emb_'+ the experiment name_. 
   3. Average scores are calculated and saved in the _preliminary experiments_ directory.
   4. Results are displayed in an interactive plot, also saved in the _preliminary experiments_ directory.

Output Sample:
![output_image](/readme_imgs/output.png)


