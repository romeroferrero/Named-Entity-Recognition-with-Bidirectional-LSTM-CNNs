import numpy as np
from keras.initializers import RandomUniform
from keras.layers import (LSTM, Bidirectional, Conv1D, Dense, Dropout,
                          Embedding, Flatten, Input, MaxPooling1D,
                          TimeDistributed, concatenate)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Progbar

from prepro import (add_char_information, create_batches, create_matrices,
                    iterate_minibatches, padding, readfile)
from validation import compute_f1

epochs = 1


trainSentences = readfile("data/train.txt")
devSentences = readfile("data/valid.txt")
testSentences = readfile("data/test.txt")

trainSentences = add_char_information(trainSentences)
devSentences = add_char_information(devSentences)
testSentences = add_char_information(testSentences)

labelSet = set()
words = {}

for dataset in [trainSentences, devSentences, testSentences]:
    for sentence in dataset:
        for token, char, label in sentence:
            labelSet.add(label)
            words[token.lower()] = True

# :: Create a mapping for the labels ::
label2idx = {}
for label in labelSet:
    label2idx[label] = len(label2idx)

# :: Hard coded case lookup ::
case2idx = {
    "numeric": 0,
    "allLower": 1,
    "allUpper": 2,
    "initialUpper": 3,
    "other": 4,
    "mainly_numeric": 5,
    "contains_digit": 6,
    "PADDING_TOKEN": 7,
}
caseEmbeddings = np.identity(len(case2idx), dtype="float32")


# :: Read in word embeddings ::
word2idx = {}
word_embeddings = []

fEmbeddings = open("embeddings/glove.6B.50d.txt", encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]

    if len(word2idx) == 0:  # Add padding+unknown
        word2idx["PADDING_TOKEN"] = len(word2idx)
        vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
        word_embeddings.append(vector)

        word2idx["UNKNOWN_TOKEN"] = len(word2idx)
        vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
        word_embeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        word_embeddings.append(vector)
        word2idx[split[0]] = len(word2idx)

word_embeddings = np.array(word_embeddings)

char2idx = {"PADDING": 0, "UNKNOWN": 1}
for (
    c
) in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2idx[c] = len(char2idx)

train_set = padding(
    create_matrices(trainSentences, word2idx, label2idx, case2idx, char2idx)
)
dev_set = padding(
    create_matrices(devSentences, word2idx, label2idx, case2idx, char2idx)
)
test_set = padding(
    create_matrices(testSentences, word2idx, label2idx, case2idx, char2idx)
)

idx2Label = {v: k for k, v in label2idx.items()}
np.save("models/idx2Label.npy", idx2Label)
np.save("models/word2idx.npy", word2idx)

train_batch, train_batch_len = create_batches(train_set)
dev_batch, dev_batch_len = create_batches(dev_set)
test_batch, test_batch_len = create_batches(test_set)


words_input = Input(shape=(None,), dtype="int32", name="words_input")
words = Embedding(
    input_dim=word_embeddings.shape[0],
    output_dim=word_embeddings.shape[1],
    weights=[word_embeddings],
    trainable=False,
)(words_input)
casing_input = Input(shape=(None,), dtype="int32", name="casing_input")
casing = Embedding(
    output_dim=caseEmbeddings.shape[1],
    input_dim=caseEmbeddings.shape[0],
    weights=[caseEmbeddings],
    trainable=False,
)(casing_input)
character_input = Input(
    shape=(
        None,
        52,
    ),
    name="char_input",
)
embed_char_out = TimeDistributed(
    Embedding(
        len(char2idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)
    ),
    name="char_embedding",
)(character_input)
dropout = Dropout(0.5)(embed_char_out)
conv1d_out = TimeDistributed(
    Conv1D(kernel_size=3, filters=30, padding="same", activation="tanh", strides=1)
)(dropout)
maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words, casing, char])
output = Bidirectional(
    LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25)
)(output)
output = TimeDistributed(Dense(len(label2idx), activation="softmax"))(output)
model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")
model.summary()
# plot_model(model, to_file='model.png')


for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch, epochs))
    a = Progbar(len(train_batch_len))
    for i, batch in enumerate(iterate_minibatches(train_batch, train_batch_len)):
        labels, tokens, casing, char = batch
        model.train_on_batch([tokens, casing, char], labels)
        a.update(i)
    a.update(i + 1)
    print(" ")

model.save("models/model.h5")

#   Performance on dev dataset
predLabels, correctLabels = tag_dataset(dev_batch)
pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_batch)
pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
