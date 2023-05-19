import os
import prelim_helpers
from PIL import ImageFont

from clip_retrieval import clip_inference

import numpy as np
from pathlib import Path
import pandas as pd
import operator
from sklearn.metrics.pairwise import cosine_similarity

wiki_wrds = prelim_helpers.load_wiki(os.path.normpath(os.getcwd() + os.sep + os.pardir) +
                                     "/datasets/data/wiktionary_wrds.txt")


def run_prelim_exp(exp_name, dummy_run=False):

    valid_exp_names = {"experiment 1", "experiment 2","experiment 3", "experiment 4"}
    if exp_name not in valid_exp_names:
        raise ValueError("Preliminary Experiment Name must be one of %r." % valid_exp_names)

    if exp_name in ("experiment 3", "experiment 4"):
        no_of_imgs_per_word = 5
    else:
        no_of_imgs_per_word = 10

    if exp_name == "experiment 4":
        #  Generate Text Embeddings
        from sentence_transformers import SentenceTransformer

        scrambled_words = prelim_helpers.scrambled_dict(wiki_wrds)
        scrambled_word_list = [scrambled_words[word] for word in wiki_wrds if len(scrambled_words[word]) == 5]
        scrambled_word_list = [word for word_list in scrambled_word_list for word in word_list]

        for model_name in ["clip-ViT-B-32", 'google/byt5-small', 'google/byt5-large']:
            print("Loading Text Embeddings for Model: %s" % model_name)
            model = SentenceTransformer(model_name)
            text_embeddings = model.encode(scrambled_word_list)
            cos_sim = cosine_similarity(text_embeddings[::no_of_imgs_per_word], text_embeddings)
            np.save("%s_%s_similarity_scores.npy" % (model_name.split("/")[-1], exp_name), cos_sim)

            plot_data = prelim_helpers.calc_avg_sims(exp_name, cos_sim, no_of_imgs_per_word)
    else:
        # Create Directories
        DIR_PREP = exp_name
        embedding_folder = "clip_emb_" + exp_name

        if not os.path.exists(DIR_PREP):
            os.makedirs(DIR_PREP)
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)

        # Experiment Prep
        if exp_name == "experiment 2":
            print("Downloading Background image.")
            prelim_helpers.download_random_img(wiki_wrds)
        elif exp_name == "experiment 3":
            print("Scrambling Words.")
            scrambled_words = prelim_helpers.scrambled_dict(wiki_wrds)

        # Create Images
        texts = []
        count = 0
        for wrd in wiki_wrds:

            if count == 100 and dummy_run == True:
                break
            if count % 500 == 0:
                print("Created %s images." % count)

            caption = wrd.upper()
            clip_caption = "an image of the word \"%s\"" % (caption)

            if exp_name in ("experiment 1", "experiment 2"):
                for font_size in range(1, 50, 5):
                    myFont = ImageFont.truetype(
                        os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/FontsFree-Net-arial-bold.ttf", font_size)
                    myImage = prelim_helpers.create_image('bg_img.jpg' if exp_name == "experiment 2" else None,
                                                          (256, 256),
                                                          'white', caption, myFont, 'black')
                    myImage.save("%s/%s_%s.jpg" % (DIR_PREP, wrd, font_size))

                    with open("%s/%s_%s.txt" % (DIR_PREP, wrd, font_size), "w") as text_file:
                        text_file.write(clip_caption)

                    texts = texts + [clip_caption]
                    count += 1
            elif exp_name == "experiment 3":  # experiment 3
                scrambled_word_list = scrambled_words[wrd]
                if len(scrambled_word_list) == 5:
                    for i_s, scram_wrd in enumerate(scrambled_word_list):
                        myFont = ImageFont.truetype(
                            os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/FontsFree-Net-arial-bold.ttf", 30)
                        myImage = prelim_helpers.create_image(None, (256, 256), 'white', scram_wrd, myFont, 'black')
                        myImage.save("%s/%s_%s.jpg" % (DIR_PREP, wrd, i_s))

                        with open("%s/%s_%s.txt" % (DIR_PREP, wrd, i_s), "w") as text_file:
                            text_file.write(clip_caption)

                        count += 1

        print("Saved %s files in folder: %s" % (len(os.listdir(DIR_PREP)), DIR_PREP))

        #  Create CLIP Embeddings
        clip_inference(input_dataset=DIR_PREP, output_folder=embedding_folder)
        print("Saved CLIP embeddings in folder: %s" % embedding_folder)

        # Load CLIP Embeddings

        text_features = np.load(embedding_folder + "/text_emb/" + "text_emb_0.npy")
        image_features = np.load(embedding_folder + "/img_emb/" + "img_emb_0.npy")
        data_dir = Path(embedding_folder + "/metadata")
        df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in data_dir.glob('*.parquet')
        )
        image_list = df["image_path"].tolist()

        # Calculate Cosine Similarity Scores

        image_info = []
        for i, img_name in enumerate(image_list):
            image_info.append(
                [image_features[i], img_name, img_name.split("_")[-2],
                 int(img_name.split("_")[-1].replace(".jpg", ""))])

        image_info = sorted(image_info, key=operator.itemgetter(2, 3))
        image_features_sorted = [im[0] for im in image_info]
        cos_sim = cosine_similarity(text_features[::no_of_imgs_per_word], image_features_sorted)
        np.save("%s_similarity_scores.npy" % exp_name, cos_sim)

    # Calculate Avg Scores
    if exp_name == "experiment 4":
        y_err_list=[]
        sim_scores_list=[]
        for model_name in ["clip-ViT-B-32", 'google/byt5-small', 'google/byt5-large']:
            cos_sim = np.load("%s_%s_similarity_scores.npy" % (model_name.split("/")[-1], exp_name))
            plot_data = prelim_helpers.calc_avg_sims(exp_name, cos_sim, no_of_imgs_per_word)
            y_err_list.append(plot_data['y_err'])
            sim_scores_list.append(plot_data['avg_sim_by_font_size_list'])
        plot_data['y_err_list'] = y_err_list
        plot_data['sim_scores_list'] = sim_scores_list
    else:
        plot_data = prelim_helpers.calc_avg_sims(exp_name, cos_sim, no_of_imgs_per_word)

    # Plot

    prelim_helpers.plot_prelim_exp(exp_name, plot_data)
