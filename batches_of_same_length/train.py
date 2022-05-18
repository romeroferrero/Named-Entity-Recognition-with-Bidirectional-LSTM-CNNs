import argparse
import logging
import os

import numpy as np
import pandas as pd
from sagemaker_training import environment
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    MaxPooling1D,
    TimeDistributed,
    concatenate,
)

from prepro import (
    add_char_information,
    create_batches,
    create_matrices,
    iterate_minibatches,
    padding,
    readfile,
    tag_dataset,
)
from validation import compute_f1

def plot_classes_counts(classes_array, class_weight=None):
    classes, counts = np.unique(classes_array, return_counts = True)
    total = np.sum(counts)
    num_classes = len(classes)
    classes_counts = {class_: count for class_, count in zip(classes, counts)}
#     plt.bar(classes, counts)
# #     print(classes_counts)
    if class_weight is None:
        class_weight = {class_: (1 / count) * (total / num_classes) for class_, count in zip(classes, counts)}
    classes_weights = np.vectorize(class_weight.get)(classes_array)
#     print(class_weight)
    return np.squeeze(classes_weights), class_weight # shape for keras when sampel_weight_complie = 'temporal' must be 2D see below when model.compile() call

def summary(model):
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    return "\n".join(summary)

if __name__ == "__main__":

    env = environment.Environment()
    job_name = env["job_name"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding_size", type=int, default=50, choices=[50, 100, 200, 300]
    )
    parser.add_argument("--num_training_samples", type=int, default=10000000)
    parser.add_argument("--num_val_samples", type=int, default=10000000)
    parser.add_argument(
        "--gpu-count", type=int, default=os.environ["SM_NUM_GPUS"]
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--training_data_file",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
    )
    parser.add_argument(
        "--validation_data_file",
        type=str,
        default=os.environ["SM_CHANNEL_VALIDATION"],
    )
    parser.add_argument(
        "--test_data_file", type=str, default=os.environ["SM_CHANNEL_TEST"]
    )
    parser.add_argument(
        "--embeddings_folder",
        type=str,
        default=os.environ["SM_CHANNEL_EMBEDDINGS"],
    )
    args, _ = parser.parse_known_args()

    print(args)

    train_sentences = readfile(args.training_data_file + '/train.txt')
    val_sentences = readfile(args.validation_data_file + '/val.txt')
    test_sentences = readfile(args.test_data_file + '/test.txt')
    print(f"{len(train_sentences)} training sentences")
    print(f"{len(val_sentences)} validation sentences")
    print(f"{len(test_sentences)} test sentences")
    print(train_sentences[0])

    train_sentences = add_char_information(train_sentences)
    val_sentences = add_char_information(val_sentences)
    test_sentences = add_char_information(test_sentences)
    all_sentences = train_sentences + val_sentences + test_sentences

    sentences_lens = [len(sentence) for sentence in all_sentences]
    max_sentence_len = max(sentences_lens)
    print(f"Max sentence lenght {max_sentence_len}")

    word_lens = [
        len(word[0]) for sentence in all_sentences for word in sentence
    ]
    max_word_len = max(word_lens)
    logging.info(f"Max word lenght {max_word_len}")

    classes = set()
    words = {}

    for sentence in all_sentences:
        for word, char, label in sentence:
            classes.add(label)
            words[word.lower()] = True

    print(f"{len(classes)} classes")
    print(f"{len(words)} words")

    # Create a mapping for the classes
    class2idx = {"PADDING_CLASS": 0}
    for label in classes:
        class2idx[label] = len(class2idx)
    print(f"{len(class2idx)} classes after adding padding")
    print(f"class2idx {class2idx}")

    # Create reverse mapping for classes and store it
    idx2class = {v: k for k, v in class2idx.items()}
    np.save("models/idx2Label.npy", idx2class)
    print(f"idx2class {idx2class}")

    # Create a look-up table for the cases
    case2idx = {
        "PADDING_WORD": 0,
        "numeric": 1,
        "allLower": 2,
        "allUpper": 3,
        "initialUpper": 4,
        "other": 5,
        "mainly_numeric": 6,
        "contains_digit": 7,
    }
    case_embeddings = np.identity(len(case2idx), dtype="float32")
    print(f"{len(case2idx)} casing types")
    print(f"case_embeddings.shape {case_embeddings.shape}")
    print(f"case2idx {case2idx}")

    embeddings_size = 50
    vocabulary_embeddings = open(
        f"{args.embeddings_folder}/glove.6B.{args.embedding_size}d.txt",
        encoding="utf-8",
    )
    # Get word embedding
    word2idx = {}
    word_embeddings = []

    # Zero verctor for padding words
    word2idx["PADDING_WORD"] = len(word2idx)
    vector = np.zeros(embeddings_size)  # Zero vector vor 'PADDING' word
    word_embeddings.append(vector)

    # Random vector for uknown words
    word2idx["UNKNOWN_WORD"] = len(word2idx)
    vector = np.random.uniform(-0.25, 0.25, embeddings_size)
    word_embeddings.append(vector)

    for line in vocabulary_embeddings:
        split = line.strip().split(" ")
        word = split[0]
        embedding = split[1:]

        if word.lower() in words:
            vector = np.array([float(num) for num in embedding])
            word_embeddings.append(vector)
            word2idx[word] = len(word2idx)

    word_embeddings = np.array(word_embeddings)

    print(f"{len(word2idx)} words in embedding")
    print(f"Words embedding shape is {word_embeddings.shape}")

    # Store word embedding indices
    # TODO: Need to indicate where th save these index in S3
    #     np.save("models/word2idx.npy", word2idx)

    chars = set()
    for sentence in all_sentences:
        for word in sentence:
            for char in word[0]:
                chars.add(char)
    print(f"{len(chars)} characters in dataset")
    all_chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
    print(f"{len(all_chars)} characters in embedding")

    # Character embedding
    char2idx = {"PADDING_CHAR": 0, "UNKNOWN_CHAR": 1}
    for c in all_chars:
        char2idx[c] = len(char2idx)
    print(
        f"{len(char2idx)} characters in after adding PADDING_CHAR and UNKNOWN_CHAR"
    )

    train_set = padding(
        create_matrices(
            train_sentences, word2idx, class2idx, case2idx, char2idx
        ),
        max_sentence_len,
        max_word_len,
        word2idx,
        case2idx,
        char2idx,
        class2idx,
    )
    val_set = padding(
        create_matrices(val_sentences, word2idx, class2idx, case2idx, char2idx),
        max_sentence_len,
        max_word_len,
        word2idx,
        case2idx,
        char2idx,
        class2idx,
    )
    test_set = padding(
        create_matrices(
            test_sentences, word2idx, class2idx, case2idx, char2idx
        ),
        max_sentence_len,
        max_word_len,
        word2idx,
        case2idx,
        char2idx,
        class2idx,
    )
    print(f"num train sentences: {len(train_set)}")
    print(f"num val sentences: {len(val_set)}")
    print(f"num test sentences: {len(test_set)}")

    train_words = np.asarray([sentence[0] for sentence in train_set])
    train_casing = np.asarray([sentence[1] for sentence in train_set])
    train_chars = np.asarray([sentence[2] for sentence in train_set])
    train_classes = np.expand_dims(
        np.asarray([sentence[3] for sentence in train_set]), -1
    )
    print(f"train_words shape {train_words.shape}")
    print(f"train_casing shape {train_casing.shape}")
    print(f"train_chars info {train_chars.shape}")
    print(f"train_classes {train_classes.shape}")
    train_sample_weights, class_weight = plot_classes_counts(train_classes)

    val_words = np.asarray([sentence[0] for sentence in val_set])
    val_casing = np.asarray([sentence[1] for sentence in val_set])
    val_chars = np.asarray([sentence[2] for sentence in val_set])
    val_classes = np.expand_dims(
        np.asarray([sentence[3] for sentence in val_set]), -1
    )
    print(f"val_words shape {val_words.shape}")
    print(f"val_casing shape {val_casing.shape}")
    print(f"val_chars shape {val_chars.shape}")
    print(f"val_classes shape {val_classes.shape}")
    val_sample_weights, class_weight = plot_classes_counts(val_classes)

    test_words = np.asarray([sentence[0] for sentence in test_set])
    test_casing = np.asarray([sentence[1] for sentence in test_set])
    test_chars = np.asarray([sentence[2] for sentence in test_set])
    test_classes = np.expand_dims(
        np.asarray([sentence[3] for sentence in test_set]), -1
    )
    print(f"{test_words.shape}")
    print(test_casing.shape)
    print(test_chars.shape)
    print(test_classes.shape)
    test_sample_weights, class_weight = plot_classes_counts(test_classes)

    # Word features come from glove.6B embeddings
    words_input = Input(
        shape=(None,), dtype="int32", name="words_input"
    )  # None indicates a variables input (sentence length is variable)
    words = Embedding(
        input_dim=word_embeddings.shape[0],
        output_dim=word_embeddings.shape[1],
        weights=[word_embeddings],
        trainable=False,
    )(words_input)

    # Casing features
    casing_input = Input(
        shape=(None,), dtype="int32", name="casing_input"
    )  # None indicates a variables input (sentence length is variable)
    casing = Embedding(
        input_dim=case_embeddings.shape[0],
        output_dim=case_embeddings.shape[1],
        weights=[case_embeddings],
        trainable=False,
    )(casing_input)

    # Character features learned with a 1D CNN
    character_input = Input(
        shape=(
            None,
            max_word_len,
        ),
        name="char_input",
    )
    embed_char_out = TimeDistributed(
        Embedding(
            input_dim=len(char2idx),
            output_dim=30,
            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
        ),
        name="char_embedding",
    )(character_input)
    dropout = Dropout(0.5)(embed_char_out)
    conv1d_out = TimeDistributed(
        Conv1D(
            kernel_size=3,
            filters=30,
            padding="same",
            activation="tanh",
            strides=1,
        )
    )(dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(max_word_len))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)

    # Concatenate features
    output = concatenate([words, casing, char])

    # Bidirectional LSTM
    output = Bidirectional(
        LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25)
    )(output)

    # Same fully connected for each word in the sequence
    output = TimeDistributed(Dense(len(class2idx), activation="softmax"))(
        output
    )

    # Define model input and output
    model = Model(
        inputs=[words_input, casing_input, character_input], outputs=[output]
    )
    # Uncomment to train with sample weights
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=['accuracy'], sample_weight_mode='temporal')
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="nadam",
        metrics=["accuracy"],
    )
    #     model.summary()

    print(summary(model))

    checkpoint_path = "/opt/ml/checkpoints"
    try:
        os.mkdir(f"{checkpoint_path}/{job_name}")
    except OSError:
        print(job_name)
        print("Creation of the directory %s failed" % checkpoint_path)
    MODEL_CHECKPOINT_FILE = f"{checkpoint_path}/model.h5"
    checkpointer = ModelCheckpoint(
        filepath=MODEL_CHECKPOINT_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

#     filename = f"{args.model_dir}/history.csv"
#     history_logger = CSVLogger(filename, separator=",", append=False)

    if args.num_training_samples is not None:
        # Note that if args.num_training_samples > len(train_words) all data will be selected
        train_words = train_words[:args.num_training_samples]
        train_chars = train_chars[:args.num_training_samples]
        train_casing = train_casing[:args.num_training_samples]
        train_classes = train_classes[:args.num_training_samples]

    if args.num_val_samples < len(val_words):
        # Note that if args.num_val_samples > len(train_words) all data will be selected
        val_words = val_words[:args.num_val_samples]
        val_casing = val_casing[:args.num_val_samples]
        val_chars = val_chars[:args.num_val_samples]
        val_classes = val_classes[:args.num_val_samples]

    print(len(train_words))
    history = model.fit(
        [train_words, train_casing, train_chars],
        train_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=[checkpointer], #, history_logger],
        validation_data=([val_words, val_casing, val_chars], val_classes),
        verbose=2,
    )
