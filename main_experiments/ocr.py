import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR

import plotly.graph_objects as go
from main_experiments.text_model import load_text_model
from main_experiments.unet_setup import UNet_SD, marginal_prob_std_fn, diffusion_coeff_fn
from main_experiments.sampler import Euler_Maruyama_sampler

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from einops import rearrange
import math
from collections import OrderedDict
from easydict import EasyDict as edict
import os
import wandb
import torchvision.transforms as T

import easyocr as ocr  # OCR
import numpy as np  # Image Processing
import random


def load_encoder_state(text_prompt, tokenizer, text_encoder, textmodel_maxtokens):
    sample_text = text_prompt
    sample_text_y = [str(sample_text)]
    sample_inputs = tokenizer(sample_text_y, max_length=textmodel_maxtokens, padding="max_length",
                              truncation=True, return_tensors="pt")
    with torch.no_grad():
        sample_encoder_hidden_states = text_encoder(sample_inputs.input_ids.cuda())[0]
        sample_encoder_hidden_states = sample_encoder_hidden_states.cpu()

    return sample_encoder_hidden_states


def run_ocr(ocr_configs):
    def load_model():
        reader = ocr.Reader(['en'], model_storage_directory='.')
        return reader

    reader = load_model()  # load model
    final_scores_dict = {}

    # Loading test words JSON file
    with open("data/datasets/mc4_test_dict.json") as json_file:
        mc4_buckets = json.loads(json_file.read())
    wiki_wrds_sample = []
    for wrd_bucket in mc4_buckets.keys():
        wiki_wrds_sample += mc4_buckets[wrd_bucket]

    model_name = ocr_configs['model_name']
    pretrained_model_name_or_path = ocr_configs['model_card']
    pretrained_model_name_or_path_2 = ocr_configs['pretrained_model_name_or_path_2']
    ckpt_path = ocr_configs['model_saved_path']

    # load required models
    transform = T.ToPILImage()
    tokenizer, text_encoder = load_text_model(pretrained_model_name_or_path)
    context_dim = list(text_encoder.named_parameters())[0][1].shape[1]
    context_dim_2 = 0
    # combined
    if ocr_configs["pretrained_model_name_or_path_2"] != "None":
        tokenizer_2, text_encoder_2 = load_text_model(pretrained_model_name_or_path_2)
        context_dim_2 = list(text_encoder_2.named_parameters())[0][1].shape[1]

    checkpoint = torch.load(ckpt_path, map_location=ocr_configs["device"])
    score_model = torch.nn.DataParallel(
        UNet_SD(marginal_prob_std=marginal_prob_std_fn, context_dim=context_dim + context_dim_2))
    score_model = score_model.to(ocr_configs["device"])
    score_model.load_state_dict(checkpoint['model'])
    score_model.eval()

    if ocr_configs["pretrained_model_name_or_path_2"] != "None":
        run_name = pretrained_model_name_or_path + pretrained_model_name_or_path_2
    else:
        run_name = pretrained_model_name_or_path

    wandb.init(project="ocr-testing", name=run_name)

    # run ocr by generating 100 samples for each word in training set:
    exact_match_count = []
    total_count = []
    result_texts = {}
    samples_dict = {}

    for word_count, word in enumerate(wiki_wrds_sample):

        sample_batch_size = 100
        num_steps = 250
        sampler = Euler_Maruyama_sampler

        text_y = "a black and white image of the word \"%s\"" % (word.upper())

        encoder_hidden_states = load_encoder_state(text_y, tokenizer, text_encoder,
                                                   ocr_configs["textmodel_maxtokens"])
        if ocr_configs['pretrained_model_name_or_path_2'] != "None":
            encoder_hidden_states_2 = load_encoder_state(text_y, tokenizer_2, text_encoder_2,
                                                         ocr_configs["textmodel_maxtokens"])
            encoder_hidden_states = torch.cat((encoder_hidden_states, encoder_hidden_states_2), dim=2)

        samples = sampler(score_model,
                          marginal_prob_std_fn,
                          diffusion_coeff_fn,
                          sample_batch_size,
                          num_steps=num_steps,
                          device=ocr_configs["device"],
                          y=encoder_hidden_states)

        samples = samples.clamp(0.0, 1.0)

        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        fig = plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        plt.show()

        wandb.log({"plot": fig})

        # run ocr model
        result_text = []  # empty list for results

        for i in range(len(samples)):
            img = transform(samples[i])
            result = reader.readtext(np.array(img))
            for text in result:
                result_text.append(text[1])

        # score results
        pos_count = 0
        for txt in result_text:
            if txt.upper() == word.upper():
                pos_count += 1

        wandb.log({'exact_match': pos_count, 'word': word.upper()})
        exact_match_count.append(pos_count)

        total_count.append(len(samples))
        result_texts[word.upper()] = result_text
        print("{} {} : EasyOCR Exact Matches for word {}: {}" \
              .format(pretrained_model_name_or_path,
                      word_count,
                      word.upper(),
                      pos_count * 100 / len(samples)))

        samples_dict[word.upper()] = samples.cpu().numpy().tolist()

    final_scores_dict[pretrained_model_name_or_path] = {}
    final_scores_dict[pretrained_model_name_or_path]['exact_match'] = exact_match_count
    final_scores_dict[pretrained_model_name_or_path]['total_count'] = total_count
    final_scores_dict[pretrained_model_name_or_path]['result_texts'] = result_texts
    final_scores_dict[pretrained_model_name_or_path]['samples'] = samples_dict

    with open('ocr_results' + model_name + '.json', 'w') as fp:
        json.dump(final_scores_dict[pretrained_model_name_or_path], fp)

    # log overall scores
    no_of_words = int(len(wiki_wrds_sample) / 5)
    mod_score = sum(final_scores_dict[pretrained_model_name_or_path]['exact_match']) / (no_of_words * 5 * 100)
    wandb.log({'overall_score': mod_score})
    print("Score for %s: %f" % (pretrained_model_name_or_path, mod_score * 100))

    # log per bucket scores
    bucket_start = 0
    for i in range(5):
        mod_score = sum(final_scores_dict[pretrained_model_name_or_path]['exact_match'][
                        bucket_start:bucket_start + no_of_words]) / (no_of_words * 100)
        std_dev_score = np.std(
            final_scores_dict[pretrained_model_name_or_path]['exact_match'][bucket_start:bucket_start + no_of_words])
        wandb.log({'mean_bucket_' + str(i): mod_score * 100, 'stdev_bucket_' + str(i): std_dev_score})
        bucket_start += no_of_words

    wandb.finish()

    print("OCR results saved in %s" % ('ocr_results' + model_name + '.json'))


def ocr_plot(ocr_plot_configs):

    # load data
    final_scores_dict = {}
    model_names = ocr_plot_configs["model_names"]
    for i, json_file in enumerate(ocr_plot_configs["model_paths"]):
        with open(json_file) as json_file2:
            data = json.load(json_file2)
        final_scores_dict[model_names[i]] = data

    # buckets
    with open('data/datasets/mc4_test_dict.json') as json_file:
        mc4_buckets = json.loads(json_file.read())

    wiki_wrds_sample = []
    for wrd_bucket in mc4_buckets.keys():
        wiki_wrds_sample += mc4_buckets[wrd_bucket]

    no_of_words = int(len(wiki_wrds_sample) / 5)

    # scores for each model
    import numpy as np
    mean_stddev_dict = {}
    for mod in final_scores_dict.keys():
        mean_stddev_dict[mod] = {'mean': [], 'std dev': []}
        bucket_start = 0
        for i in range(5):
            #         print(bucket_start)
            mod_score = sum(final_scores_dict[mod]['exact_match'][bucket_start:bucket_start + no_of_words]) / (
                    no_of_words * 100)
            #         std_dev_score = np.std(final_scores_dict[mod]['exact_match'][bucket_start:bucket_start+no_of_words])
            std_dev_score = 196 * np.sqrt((mod_score * (1 - mod_score)) / (no_of_words * 100))
            # print("Score for Bucket %d %s: %f, %f" % (i, mod, mod_score * 100, std_dev_score))
            mean_stddev_dict[mod]['mean'] += [mod_score * 100]
            mean_stddev_dict[mod]['std dev'] += [std_dev_score]

            #         print(bucket_start,bucket_start+17,len(final_scores_dict[mod]['exact_match'][bucket_start:bucket_start+17]))
            bucket_start += no_of_words
    # plot

    fig = go.Figure()
    encoder_colours = ['#bad5db', '#7cc489', '#215434']

    for i_m, mod in enumerate(final_scores_dict.keys()):
        fig.add_trace(go.Bar(
            name= model_names[i_m],
            x=["Top 1%","1-10%","10-20%","20-30%","Bottom 50%"], y=mean_stddev_dict[mod]['mean'],
            error_y=dict(type='data', array=mean_stddev_dict[mod]['std dev']),
            error_y_thickness=1,
            marker_color=encoder_colours[i_m]
        ))

    fig.update_layout(template='plotly_white',
                      font=dict(
                          family="Arial Narrow",
                          size=13
                      ),
                      title_text='<b>OCR Results</b>',
                      title_x=0.5,
                      xaxis_title="<b>Word Frequency</b>",
                      yaxis_title="<b>% Exact Matches</b>",
                      legend_title="",
                      bargroupgap=0.05,
                      #     yaxis_range=[0,100]

                      )
    fig.show()
