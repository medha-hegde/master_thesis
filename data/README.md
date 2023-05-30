## Data for Master Thesis "Improving Text Rendering in Image Generation"

This repository contains the datasets and related code required to run the experiments.

The _datasets_ directory contains the following files, all created using the create_wiki_dataset.py file:

1. **wiktionary_wrds.txt**: Preprocessed Wiktionary Words ([Ylonen, 2022](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.140.pdf)) filtered to only single words, length<30 characters, no punctuation or symbols. This list of words is used in the Preliminary Experiments and the Main Experiments.
2. **mc4_data.txt**: List of all words in train set for Main Experiments
3. **mc4_train_dict.json**: Dictionary of words in train set for Main Experiments, grouped by word frequency bucket
4. **mc4_test_dict.json**: Dictionary of words in test set for Main Experiments, grouped by word frequency bucket




