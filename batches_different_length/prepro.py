import random

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Progbar


def readfile(filename):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'],
    ['rejects', 'O'],
    ['German', 'B-MISC'],
    ['call', 'O'], ['to', 'O'],
    ['boycott', 'O'],
    ['British', 'B-MISC'],
    ['lamb', 'O'],
    ['.', 'O'] ]
    """
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(" ")
        sentence.append([splits[0], splits[-1]])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences


def get_casing(word, case_lookup):
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


def create_batches(sentences):
    sentence_lenghts = []
    for sentence in sentences:
        sentence_lenghts.append(len(sentence[0]))
    unique_lengths = set(sentence_lenghts)
    batches = []
    num_sentences_per_batch = []
    sentence_count = 0
    for length in unique_lengths:
        for sentence in sentences:
            if len(sentence[0]) == length:
                batches.append(sentence)
                sentence_count += 1
        num_sentences_per_batch.append(sentence_count)
    return unique_lengths, batches, num_sentences_per_batch


def create_matrices(sentences, word2idx, label2idx, case2idx, char2idx):
    unknown_idx = word2idx["UNKNOWN_TOKEN"]
    padding_idx = word2idx["PADDING_TOKEN"]

    dataset = []

    word_count = 0
    unknown_word_count = 0

    for sentence in sentences:
        word_idxs = []
        case_idxs = []
        char_idxs = []
        class_idxs = []

        for word, char, label in sentence:
            word_count += 1
            if word in word2idx:
                word_idx = word2idx[word]
            elif word.lower() in word2idx:
                word_idx = word2idx[word.lower()]
            else:
                word_idx = unknown_idx
                unknown_word_count += 1
            char_idx = []
            for x in char:
                char_idx.append(char2idx[x])
            # Get the label and map to int
            word_idxs.append(word_idx)
            case_idxs.append(get_casing(word, case2idx))
            char_idxs.append(char_idx)
            class_idxs.append(label2idx[label])

        dataset.append([word_idxs, case_idxs, char_idxs, class_idxs])

    return dataset

def _prepare_batch(sentences):
    all_words = []
    all_caseings = []
    all_characters = []
    all_classes = []
    for sentence in sentences:
        words, words_casing, words_characters, words_classes = sentence
        words_classes = np.expand_dims(words_classes, -1)
        all_words.append(words)
        all_caseings.append(words_casing)
        all_characters.append(words_characters)
        all_classes.append(words_classes)
    return np.asarray(all_classes), np.asarray(all_words), np.asarray(all_caseings), np.asarray(all_characters)


def iterate_minibatches(all_sentences, batch_len):
    start = 0
    for i in batch_len:
        sentences_of_same_length = all_sentences[start:i]
        start = i
        yield _prepare_batch(sentences_of_same_length)


def add_char_information(sentences):
    for i, sentence in enumerate(sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            sentences[i][j] = [data[0], chars, data[1]]
    return sentences


def padding(sentences):
    maxlen = 52
    for sentence in sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(sentences):
        sentences[i][2] = pad_sequences(sentences[i][2], 52, padding="post")
    return sentences


def tag_dataset(model, dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing, char], verbose=False)[0]
        pred = pred.argmax(axis=-1)  # Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i + 1)
    return predLabels, correctLabels
