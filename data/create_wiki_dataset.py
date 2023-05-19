from json import JSONDecodeError
import regex

from datasets import load_dataset
import nltk
import re
import json
from random import sample, seed, shuffle


def wiktionary_preproc(path_to_wiki_json):
    # create list of all words
    wrd_lst = []
    with open(path_to_wiki_json, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                wrd_lst.append(data['word'])
            except JSONDecodeError as e:
                print(e)

    # filter out punctuation, symbols, and words longer than 30 characters
    flt_wrd_lst = []
    for wrd in wrd_lst:
        if " " not in wrd:
            punc = regex.search("\p{P}", wrd)
            symb = regex.search("\p{S}", wrd)
            if punc is None and symb is None:
                if len(wrd) < 30:
                    flt_wrd_lst.append(wrd)

    # remove repeated words
    set_wrds = set(flt_wrd_lst)
    unique_wrds = (list(set_wrds))

    # write to file
    with open('datasets/wiktionary_wrds.txt', 'w') as f:
        for line in unique_wrds:
            f.write(f"{line}\n")


def create_wiki_dataset():
    # load data
    wiki_wrds = []
    with open('datasets/wiktionary_wrds.txt') as wrd_list:
        for wrd in wrd_list:
            if len(wrd) < 10:
                wiki_wrds.append(wrd.replace('\n', '').lower())

    print("No. of Wiktionary words:", len(wiki_wrds))

    # download c4 data
    dataset_c4 = load_dataset("allenai/c4",
                              data_files=["en/c4-train.00000-of-01024.json.gz",
                                          "en/c4-train.00001-of-01024.json.gz"])

    c4_corpus = [data['text'] for data in dataset_c4['train']]
    print("No. of C4 documents: ", len(c4_corpus))

    # tokenize c4 data
    words = []
    for c4_doc in c4_corpus:
        token = re.findall('\w+', c4_doc)
        words += [word.lower() for word in token]

    print("No. of unique C4 words: ", len(words))

    # create frequency distribution of c4 words

    nlp_words = nltk.FreqDist(words)
    freq_list = nltk.FreqDist({k: nlp_words.get(k, 0) for k in wiki_wrds})

    # create buckets
    top1_n = int(len(freq_list) * .01)
    top1_10_n = int(len(freq_list) * .1)
    top10_20_n = int(len(freq_list) * .2)
    top20_30_n = int(len(freq_list) * .3)
    bottom_50_n = int(len(freq_list) * .5)

    top1_words = freq_list.most_common()[:top1_n]
    top1_10_words = freq_list.most_common()[top1_n:top1_10_n]
    top10_20_words = freq_list.most_common()[top1_10_n:top10_20_n]
    top20_30_words = freq_list.most_common()[top10_20_n:top20_30_n]
    bottom_50_words = freq_list.most_common()[bottom_50_n:]

    for i, word_bucket in enumerate([top1_words, top1_10_words, top10_20_words, top20_30_words, bottom_50_words]):
        print("No. of words in bucket %d = %d" % (i + 1, len(word_bucket)))

    # create test set

    test_set_names = ["test_top1", "test_top1_10", "test_top10_20", "test_top20_30", "test_bottom_50"]
    test_words = {}

    for i, word_bucket in enumerate([top1_words, top1_10_words, top10_20_words, top20_30_words, bottom_50_words]):
        words_only = [w[0] for w in word_bucket]
        seed(123)
        test_words[test_set_names[i]] = sample(words_only, 17)

    # create training set

    # calculate total word freq per bucket
    freq_per_bucket = []
    for i, word_bucket in enumerate([top1_words, top1_10_words, top10_20_words, top20_30_words, bottom_50_words]):
        freq_per_bucket.append(sum([w[1] for w in word_bucket]))

    # sample proportionally
    train_set_names = ["train_top1", "train_top1_10", "train_top10_20", "train_top20_30", "train_bottom_50"]
    train_words_bucket = {}
    train_words = []

    for i, word_bucket in enumerate([top1_words, top1_10_words, top10_20_words, top20_30_words]):
        n_sample = round(freq_per_bucket[i] * 3500 / sum(freq_per_bucket))
        print("Bucket %d sample size = %d words" % (i + 1, n_sample))

        # exclude test words
        words_only = [w[0] for w in word_bucket if w[0] not in test_words[test_set_names[i]]]
        # print(len(words_only))
        seed(123)
        bucket_sample = sample(words_only, n_sample)
        train_words += bucket_sample
        train_words_bucket[train_set_names[i]] = bucket_sample

    # bottom 50% sample
    bottom_50_words_only = [w[0] for w in bottom_50_words if w[0] not in test_words['test_bottom_50']]
    seed(123)
    bucket_sample = sample(bottom_50_words_only, 3500)
    train_words += bucket_sample
    train_words_bucket['train_bottom_50'] = bucket_sample
    print("Bucket 5 sample size = %d words" % (len(bucket_sample)))

    # save train list of words
    seed(123)
    shuffle(train_words)
    with open('datasets/mc4_data.txt', 'w') as tfile:
        tfile.write('\n'.join(train_words))

    # check if any overlap b/w train and test sets
    for wrd in train_words:
        for tst_set in test_set_names:
            if wrd in test_words[tst_set]:
                print("!! Word appears in both train and text set!!", wrd)

    # save test/train dict  data files
    with open('datasets/mc4_train_dict.json', 'w') as fp:
        json.dump(train_words_bucket, fp)

    with open('datasets/mc4_test_dict.json', 'w') as fp:
        json.dump(test_words, fp)
