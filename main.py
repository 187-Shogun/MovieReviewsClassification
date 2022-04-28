"""
Title: main.py

Created on: 1/6/2022

Author: 187-Shogun

Encoding: UTF-8

Description: Binary classifier to perform sentiment analysis on the IMDB dataset.
"""


from datetime import datetime
from pytz import timezone
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os
import re
import string


# Global variables to interact with the script:
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 5
RANDOM_SEED = 69
MAX_FEATURES = 25_000
SEQ_LENGTH = 128
EMBEDDING_DIM = 64
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
CM_DIR = os.path.join(os.getcwd(), 'reports')
MODELS_DIR = os.path.join(os.getcwd(), 'models')


def get_model_version_name(model_name: str) -> str:
    """ Generate a unique name using timestamps. """
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


def get_dataset() -> tuple:
    """ Get dataset from the TFDS library. """
    train_a, info = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
        split='train+test[:70%]',
        batch_size=BATCH_SIZE
    )
    train_b, val, test = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        as_supervised=True,
        split=['test[70%:80%]', 'test[80%:90%]', 'test[90%:100%]'],
        batch_size=BATCH_SIZE
    )
    # Unpack elements:
    train = train_a.concatenate(train_b)
    return train, val, test, info


def text_normalization(data) -> tf.Tensor:
    """ Clean text. """
    lowercase = tf.strings.lower(data)
    html_stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(html_stripped, '[%s]' % re.escape(string.punctuation), '')


def txt_to_vec_layer(max_features: int, sequence_length: int) -> tf.keras.layers.TextVectorization:
    """ Create a TextVectorization layer to transform each word into a unique integer in an index. """
    corpus_a = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        as_supervised=True,
        split=['train'],
    )
    corpus_b = tfds.load(
        'imdb_reviews',
        shuffle_files=True,
        as_supervised=True,
        split=['unsupervised'],
    )
    vxt = tf.keras.layers.TextVectorization(
        standardize=text_normalization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    vxt.adapt(corpus_a[0].concatenate(corpus_b[0]).map(lambda x, y: x))
    return vxt


def text_to_vector(vxt_layer, sample_text, sample_label) -> tf.keras.layers.TextVectorization:
    """" Apply processing layer to input text. """
    sample_text = tf.expand_dims(sample_text, -1)
    return vxt_layer(sample_text), sample_label


def build_dummy_network() -> tf.keras.Sequential:
    """ Build a baseline network. """
    lyrs = [
        txt_to_vec_layer(max_features=MAX_FEATURES, sequence_length=SEQ_LENGTH),
        tf.keras.layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.Sequential(name='Dummy-NN', layers=lyrs)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        metrics=tf.keras.metrics.BinaryAccuracy()
    )
    return model


def build_custom_network() -> tf.keras.Sequential:
    """ Build a sequential model using a pretrained embedding layer from TFHub. """
    # TFHub Sources:
    emb_layer = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"

    # Assemble the model:
    lyrs = [
        hub.KerasLayer(emb_layer, input_shape=[], dtype=tf.string),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.Sequential(name='Pretrained-NN', layers=lyrs)

    # Compile it and return it:
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True),
        metrics=tf.keras.metrics.BinaryAccuracy()
    )
    return model


def build_custom_network_alpha() -> tf.keras.Sequential:
    """ Build a sequential model using a pretrained embedding layer from TFHub
    and implement Convolutinal layers on top of the embeddings. """
    # TFHub Sources:
    emb_layer = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"

    # Assemble the model:
    lyrs = [
        hub.KerasLayer(emb_layer, input_shape=[], dtype=tf.string),
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 128, 1))),
        tf.keras.layers.Conv1D(32, 8, strides=2, activation='relu'),
        tf.keras.layers.Conv1D(64, 16, strides=2, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.Sequential(name='Pretrained-CNN', layers=lyrs)

    # Compile it and return it:
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        metrics=tf.keras.metrics.BinaryAccuracy()
    )
    return model


def build_custom_network_sigma() -> tf.keras.Sequential:
    """ Build a sequential model using a pretrained embedding layer from TFHub
    and implement Recurrent layers on top of the embeddings. """
    # TFHub Sources:
    emb_layer = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"

    # Assemble the model:
    lyrs = [
        hub.KerasLayer(emb_layer, input_shape=[], dtype=tf.string),
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 128, 1))),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.Sequential(name='Pretrained-RNN', layers=lyrs)

    # Compile it and return it:
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True),
        metrics=tf.keras.metrics.BinaryAccuracy()
    )
    return model


# noinspection PyCallingNonCallable
def bert_network() -> tf.keras.Model:
    """ Return a pretrained BERT model from tensorflow hub. """
    # TFHub Sources:
    preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"

    # Assemble model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(preprocessor_url)
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(encoder_url)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]
    cls_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    cls_output = cls_layer(pooled_output)

    # Compile it and return it:
    model = tf.keras.Model(inputs=text_input, outputs=cls_output, name='BERT-DNN')
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        metrics=tf.keras.metrics.BinaryAccuracy()
    )
    return model


def custom_training(model, training_ds, validation_ds, pretrain_rounds=20) -> tf.keras.Model:
    """ Train a custom model in 2 rounds. """
    # Start pretraining:
    version_name = get_model_version_name(model.name)
    tb_logs = tf.keras.callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    model.fit(training_ds, validation_data=validation_ds, epochs=pretrain_rounds, callbacks=[tb_logs])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))

    # Unfreeze layers and train the entire network:
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.9, nesterov=True),
        metrics=tf.keras.metrics.BinaryAccuracy()
    )
    early_stop = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=int(PATIENCE/2))
    model.fit(training_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))
    return model


def test_trained_network() -> list:
    """ Load up an existing network and run prediction over a given dataset. """
    # Fetch datasets and configure them:
    X_train, X_val, X_test, info = get_dataset()
    X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)

    # Load up the model:
    selected_model = r"Models/Pretrained-NN_v.20220223-182329.h5"
    model = tf.keras.models.load_model(selected_model, custom_objects={'KerasLayer': hub.KerasLayer})
    scores = model.evaluate(X_test)
    print(scores)
    return scores


def dev():
    """ Entry point for testing purposes. """
    # Fetch datasets and configure them:
    X_train, X_val, X_test, info = get_dataset()
    X_train = X_train.cache().prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.cache().prefetch(buffer_size=AUTOTUNE)

    # Build a network and train it:
    model = build_custom_network_sigma()
    model.fit(X_train, validation_data=X_val, epochs=10)
    return {}


def main():
    """ Run script. """
    # Fetch datasets and configure them:
    X_train, X_val, X_test, info = get_dataset()
    X_train = X_train.cache().prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.cache().prefetch(buffer_size=AUTOTUNE)
    X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)

    # Build a network and train it:
    model = build_custom_network()
    model = custom_training(model, X_train, X_val)
    score = model.evaluate(X_test)
    print(score)
    return {}


if __name__ == '__main__':
    main()
