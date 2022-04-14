import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize


class Parser:
    def __init__(self):
        # ::Hard coded char lookup ::
        self.char2idx = {"PADDING": 0, "UNKNOWN": 1}
        for (
            c
        ) in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
            self.char2idx[c] = len(self.char2idx)
        # :: Hard coded case lookup ::
        self.case2idx = {
            "numeric": 0,
            "allLower": 1,
            "allUpper": 2,
            "initialUpper": 3,
            "other": 4,
            "mainly_numeric": 5,
            "contains_digit": 6,
            "PADDING_TOKEN": 7,
        }

    def load_models(self, loc=None):
        if not loc:
            loc = os.path.join(os.path.expanduser("~"), ".ner_model")
        self.model = load_model(os.path.join(loc, "model.h5"))
        # loading word2idx
        self.word2idx = np.load(os.path.join(loc, "word2idx.npy")).item()
        # loading idx2Label
        self.idx2Label = np.load(os.path.join(loc, "idx2Label.npy")).item()

    def get_casing(self, word, case_lookup):
        casing = "other"

        num_digits = 0
        for char in word:
            if char.isdigit():
                num_digits += 1

        digit_fraction = num_digits / float(len(word))

        if word.isdigit():  # Is a digit
            casing = "numeric"
        elif digit_fraction > 0.5:
            casing = "mainly_numeric"
        elif word.islower():  # All lower case
            casing = "allLower"
        elif word.isupper():  # All upper case
            casing = "allUpper"
        elif word[0].isupper():  # is a title, initial char upper, then all lower
            casing = "initialUpper"
        elif num_digits > 0:
            casing = "contains_digit"
        return case_lookup[casing]

    def createTensor(self, sentence, word2idx, case2idx, char2idx):
        unknown_idx = word2idx["UNKNOWN_TOKEN"]

        word_idxs = []
        case_idxs = []
        char_idxs = []

        for word, char in sentence:
            word = str(word)
            if word in word2idx:
                word_idx = word2idx[word]
            elif word.lower() in word2idx:
                word_idx = word2idx[word.lower()]
            else:
                word_idx = unknown_idx
            char_idx = []
            for x in char:
                if x in char2idx.keys():
                    char_idx.append(char2idx[x])
                else:
                    char_idx.append(char2idx["UNKNOWN"])
            word_idxs.append(word_idx)
            case_idxs.append(self.get_casing(word, case2idx))
            char_idxs.append(char_idx)

        return [word_idxs, case_idxs, char_idxs]

    def addCharInformation(self, sentence):
        return [[word, list(str(word))] for word in sentence]

    def padding(self, Sentence):
        Sentence[2] = pad_sequences(Sentence[2], 52, padding="post")
        return Sentence

    def predict(self, Sentence):
        Sentence = words = word_tokenize(Sentence)
        Sentence = self.addCharInformation(Sentence)
        Sentence = self.padding(
            self.createTensor(Sentence, self.word2idx, self.case2idx, self.char2idx)
        )
        tokens, casing, char = Sentence
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = self.model.predict([tokens, casing, char], verbose=False)[0]
        pred = pred.argmax(axis=-1)
        pred = [self.idx2Label[x].strip() for x in pred]
        return list(zip(words, pred))
