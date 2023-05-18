import random
from PIL import Image, ImageDraw
from clip_retrieval.clip_client import ClipClient
import requests
import numpy as np
import plotly.graph_objects as go
import inspect

# Load Wiktionary Words
def load_wiki(path_to_file):
    wiki_wrds_all = []
    with open(path_to_file) as wrd_list:
        for wrd in wrd_list:
            wiki_wrds_all.append(wrd.replace('\n', '').lower())
    random.seed(1234)
    return random.sample(wiki_wrds_all, 1000)


# Resize Image Function
def resize_img(img, img_size, fill_color):
    w, h = img.size[0], img.size[1]

    if w < img_size and h < img_size:  # resize image is smaller than img_size
        if w > h:
            new_w = img_size
            new_h = int(h * img_size / w)
        else:
            new_h = img_size
            new_w = int(w * img_size / h)
        img = img.resize((new_w, new_h))
    else:
        img.thumbnail((img_size, img_size),
                      Image.ANTIALIAS)  # antialias = resampling method calculated using a high-quality Lanczos filter

    w, h = img.size[0], img.size[1]
    img_prepro = Image.new('RGB', (img_size, img_size),
                           fill_color)  # create plain image w/ inputted fill colour
    try:
        img_prepro.paste(img, (int((img_size - w) / 2), int((img_size - h) / 2)), mask=img.split()[
            3])  # img.split splits the image to get the alpha (transparency) channel (to blend the edges)
    except:
        img_prepro.paste(img, (
            int((img_size - w) / 2), int((img_size - h) / 2)))  # place image in centre of fill colour image

    return img_prepro


# Print text on image
def create_image(bgImg, size, bgColor, message, font, fontColor):
    if bgImg == None:
        image = Image.new('RGB', size, bgColor)
    else:
        image = Image.open(bgImg).convert('RGB')
        image = resize_img(image, size[0], bgColor)
    draw = ImageDraw.Draw(image)
    W, H = size
    _, _, w, h = draw.textbbox((0, 0), message, font=font)
    draw.text(((W - w) / 2, (H - h) / 2), message, font=font, fill=fontColor)
    return image


# Download Random Background Image
def download_random_img(wiki_wrds):
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14")
    sample_img_words = ["orange", "cat", "lemon", "dog", "phone"]
    for sample_wrd in sample_img_words:
        if sample_wrd not in wiki_wrds:
            results = client.query(text="an image of a %s" % sample_wrd)
            break
    img_data = requests.get(results[0]['url']).content
    with open('bg_img.jpg', 'wb') as handler:
        handler.write(img_data)


# Scramble word function
def switch_n_letters(word, n):
    w = word
    if len(w) > 3 and n < (len(w) - 1):
        w = list(w)
        old_index = random.sample(list(range(1, len(w) - 1)), n)
        selected_letters = [w[i] for i in old_index]
        new_index = old_index[:]
        while True:
            random.shuffle(new_index)
            for a, b in zip(old_index, new_index):
                if a == b:
                    break
            else:
                for ind, letter in zip(new_index, selected_letters):
                    w[ind] = letter

                return ''.join(w)
    else:
        return word


# Create Scrambled Words
def scrambled_dict(wiki_wrds):
    scrambled_words = {}
    count = 0
    for word in wiki_wrds:
        count += 1
        scrambled_words[word] = [word]
        if len(word) <= 6:
            for n in range(2, len(word) - 1):
                scrambled_words[word].append(switch_n_letters(word, n))

        elif len(word) == 7:
            for n in [2, 3, 4]:
                scrambled_words[word].append(switch_n_letters(word, n))

        elif len(word) > 7:
            n_scr = []
            for pct in np.linspace(0.25, 0.75, 3):
                n_scr.append(round(pct * (len(word))))

            n_scr_set = set(n_scr)
            n_scr_set -= {0, 1, len(word)}
            n_scr = list(n_scr_set)
            n_scr.sort()
            for n in n_scr:
                scrambled_words[word].append(switch_n_letters(word, n))
        wrd_list = list(word)
        random.shuffle(wrd_list)
        scrambled_words[word].append(''.join(wrd_list))
    return scrambled_words

# Calculate Average Similarities
def calc_avg_sims(exp_name,cos_sim,no_of_imgs_per_word):

    def retrieve_name(var):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

    print("Calculating Similarity Scores!")


    similarities_by_word = []
    nonsim_by_word = []
    for txt_i in range(cos_sim.shape[0]):
        similarities = cos_sim[txt_i, txt_i * no_of_imgs_per_word:(txt_i * no_of_imgs_per_word) + no_of_imgs_per_word]
        non_similarities = np.concatenate((cos_sim[txt_i, :txt_i * no_of_imgs_per_word],
                                           cos_sim[txt_i, (txt_i * no_of_imgs_per_word) + no_of_imgs_per_word:]),
                                          dtype=object)

        similarities_by_word.append(similarities)
        nonsim_by_word.append(non_similarities)

    avg_sim_by_font_size = np.mean(similarities_by_word, axis=0)
    avg_nonsim_by_font_size_all = np.mean(nonsim_by_word, axis=0)
    avg_range_by_font_size = np.std(similarities_by_word, axis=0)
    avg_range_non = np.std(avg_nonsim_by_font_size_all, axis=0)

    plot_data = {}
    for plot_data_point in [avg_range_by_font_size, avg_range_non, avg_sim_by_font_size, avg_sim_by_font_size]:
        plot_data[retrieve_name(plot_data_point)] = plot_data_point

    if exp_name in ("experiment 3","experiment 4"):
        avg_nonsim = np.mean(avg_nonsim_by_font_size_all)
        y_err = [avg_range_by_font_size.tolist() + [avg_range_non]]
        avg_sim_by_font_size_list = avg_sim_by_font_size.tolist() + [avg_nonsim]
        plot_data['avg_nonsim'] = avg_nonsim
        plot_data['avg_sim_by_font_size_list'] = avg_sim_by_font_size_list
        plot_data['y_err'] = y_err
    else:
        avg_nonsim = []
        non_avg_range_by_font_size = []
        for i in range(no_of_imgs_per_word):
            avg_nonsim.append(np.mean(avg_nonsim_by_font_size_all[i::no_of_imgs_per_word]))
            non_avg_range_by_font_size.append(np.std(avg_nonsim_by_font_size_all[i::no_of_imgs_per_word]))
        plot_data['avg_nonsim'] = avg_nonsim
        plot_data['non_avg_range_by_font_size'] = non_avg_range_by_font_size

    return plot_data

# Plot
def plot_prelim_exp(exp_name, plot_data):
    fig = go.Figure()
    legend_names = ['CLIP', 'ByT5-small', 'ByT5-large'] if exp_name == 'experiment 4' else ['Same Word',
                                                                                            'Different Word']
    if exp_name in ("experiment 3", "experiment 4"):

        if exp_name == "experiment 4":
            bar_cols_list = []
            green_cols = ['#7cc489', '#578a60', '#34543a']
            for i, bar_col in enumerate(['#bad5db', '#81c6d6', '#476f78']):
                bar_colours = [bar_col] * len(plot_data['avg_sim_by_font_size_list'])
                bar_colours[-1] = green_cols[i]
                bar_cols_list.append(bar_colours)
        else:
            bar_colours = ['#bad5db'] * len(plot_data['avg_sim_by_font_size_list'])
            bar_colours[-1] = '#7cc489'
            # y_err = [plot_data['avg_range_by_font_size'].tolist() + [plot_data['avg_range_non']]]

        for i_m, y_value in enumerate([plot_data['avg_sim_by_font_size_list']] if exp_name == 'experiment 3' else plot_data['sim_scores_list']):
            fig.add_trace(go.Bar(
                name=legend_names[i_m],
                x=["0%", "25%", "50%", "75%", "100%", "Other Words"], y=y_value,
                error_y=dict(type='data', array=plot_data['y_err'][i_m] if exp_name == 'experiment 3' else plot_data['y_err_list'][i_m][0]),
                error_y_thickness=.5,
                marker_color=bar_colours if exp_name == 'experiment 3' else bar_cols_list[i_m],
            ))
    else:
        encoder_colours = ['#bad5db', '#7cc489']
        y_err = [plot_data['avg_range_by_font_size'], plot_data['non_avg_range_by_font_size']]

        for i_m, y_value in enumerate([plot_data['avg_sim_by_font_size'], plot_data['avg_nonsim']]):
            fig.add_trace(go.Bar(
                name=legend_names[i_m],
                x=np.arange(1, 50, 5), y=y_value,
                error_y=dict(type='data', array=y_err[i_m]),
                error_y_thickness=.5,
                marker_color=encoder_colours[i_m]
            ))

    fig.update_layout(template='plotly_white',
                      font=dict(
                          family="Arial Narrow",
                          size=13
                      ),
                      #     title_text='Experiment 1',
                      title_x=0.5,
                      xaxis_title="Scrambling Percentage" if exp_name in (
                      "experiment 3", "experiment 4") else "Font Size",
                      yaxis_title="Similarity Score" if exp_name in (
                      "experiment 3", "experiment 4") else "CLIP Score",
                      legend_title="",
                      bargroupgap=0.05,
                      #     yaxis_range=[0,0.6],

                      )

    if exp_name not in ("experiment 3", "experiment 4"):
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=5))

    fig.show()

    # save image
    fig.write_image("%s.png" % exp_name, scale=10)

    print("Plot for %s saved as %s. " % (exp_name, exp_name + ".png"))



