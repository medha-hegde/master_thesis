import torch
import os, json, random, sys
from params import params
import numpy as np
from transformers import AutoTokenizer
from torch.utils import data
import re
import string

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

do_lowercase = lambda x: x.lower() if not params.case_sensitive else x
MAX_LEN = 0

basepath = "/".join(os.path.realpath(__file__).split('/')[:-1])

if 'clip' in params.model_card:
    regex_pattern = r'[a-zA-Z]+$'
elif 't5' in params.model_card:
    regex_pattern = r'▁?[a-zA-Z]+$'
elif 'bert-' not in params.model_card:
    regex_pattern = r'Ġ?[a-zA-Z]+$'
else:
    regex_pattern = r'#*[a-zA-Z]+$'

bert_tokenizer = AutoTokenizer.from_pretrained(params.model_card)


full_dataset = sorted(bert_tokenizer.vocab.items(), key=lambda x: x[1])
full_dataset = [x for x in full_dataset if re.match(regex_pattern, x[0].replace("</w>","")) and
                len(x[0].replace('</w>', '')) > 1 and set(x[0]) != {'Ġ'}]

char_vocab = list(set([x.lower() for d in full_dataset for x in d[0]]))
print(char_vocab)
print("Len Char Vocab:", len(char_vocab))

char_to_id = {c: i for i, c in enumerate(char_vocab)}
id_to_char = {i: c for i, c in enumerate(char_vocab)}

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
clean_up = lambda x: x.replace("</w>","")

lemma_wise = {x: [] for x in set(lemmatizer.lemmatize(clean_up(x[0])) for x in full_dataset)}
for x in full_dataset:
    lemma_wise[lemmatizer.lemmatize(clean_up(x[0]))].append(x)

flatten = lambda x: [y for xx in x for y in xx]
full_dataset = list(lemma_wise.values())

# Split the dataset
random.shuffle(full_dataset)


def shuffle_and_return(x):
    random.shuffle(x)
    return x


class SpellingDataset:
    def __init__(self):
        self.bert_tokenizer = bert_tokenizer

        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        self.alphabets = string.ascii_lowercase

        self.split = int(0.8 * len(full_dataset))
        random.shuffle(full_dataset)
        train_set = flatten(full_dataset[:self.split])
        test_set = flatten(full_dataset[self.split:])

        self.alphabet_wise_datasets = {c: self.split_and_process(c, train_set, test_set)
                                       for c in self.alphabets
                                       }

    def split_and_process(self, c, train_data, test_data):
        train_data = self.balance_dataset(c, train_data)
        test_data = self.balance_dataset(c, test_data)
        if params.dummy_run:
            test_data = train_data
        return (self.process(c, train_data), self.process(c, test_data))

    def balance_dataset(self, c, train_set):
        splitted_set = ([x for x in train_set if c in x[0].replace('</w>', '')],
                        [x for x in train_set if c not in x[0].replace('</w>', '')])
        assert len(splitted_set[0]) + len(splitted_set[1]) == len(train_set)

        train_set = splitted_set[0] + splitted_set[1][:len(splitted_set[0])]
        random.shuffle(train_set)
        return train_set

    def process(self, c, all_data):
        if params.dummy_run:
            all_data = all_data[:5]
        return [(x[0], self.bert_tokenizer.convert_tokens_to_ids(x[0]),
                 int(c in do_lowercase(x[0].replace("</w>",""))))
                for x in all_data]


def pad(batch):
    get_f = lambda x: [single_data[x] for single_data in batch]
    batch_tokens = get_f(0)
    token_ids_tensor = torch.LongTensor(get_f(1)).to(params.device)
    char_ids_tensor = torch.FloatTensor(get_f(2)).to(params.device)

    return batch_tokens, token_ids_tensor, char_ids_tensor


if __name__ == "__main__":
    dataset = SpellingDataset()
    print("Num chars:", len(dataset.alphabet_wise_datasets))

    print({x[0]: len(x[1][0]) for x in dataset.alphabet_wise_datasets.items()})
    print('\n')
    print({x[0]: len(x[1][1]) for x in dataset.alphabet_wise_datasets.items()})

    print(dataset.alphabet_wise_datasets['a'][0][:5])
    print(dataset.alphabet_wise_datasets['a'][1][:5])

    print(pad(dataset.alphabet_wise_datasets['a'][0][:5]))