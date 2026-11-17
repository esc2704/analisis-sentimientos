from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from config import RESULTS_DIR
from preprocessing.prepare_dataset import preprocess

SEED = 6
EMB_DIM = 300            # tamaño del vector de cada palabra
W2V_WINDOW = 5           # palabras de contexto a cada lado, 5
W2V_MIN_COUNT = 2        # ignora palabras que aparecen menos veces de 2
MAX_LEN = 140            # No mas de 140
BATCH_SIZE = 64
EPOCHS = 10              # máximo; EarlyStopping corta antes si val_auc deja de mejorar
VAL_SIZE = 0.10          # fracción de train que reservamos para validación
ENTRENAR_EMBEDDINGS = False  # True = ajustar los vectores de word2vec durante el entrenamiento

MODELS_DIR = RESULTS_DIR / "models"


class Word2VecClassifier:

    def __init__(
        self,
        vector_size=EMB_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        sg=1,
        random_state=SEED,
        model_dir=MODELS_DIR / "word2vec",
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.random_state = random_state

        self.model_dir = Path(model_dir)
        self.w2v_path = self.model_dir / "word2vec.model"
        self.embeddings_path = self.model_dir / "embedding_matrix.npy"
        self.weights_path = self.model_dir / "classifier.weights.h5"

        self.w2v = None
        self.vectorizator = None
        self.embedding_matrix = None
        self.model = None
        self.history = None

    def _train_word2vec(self, df_train):
        # Word2Vec solo con train para no filtrar información de val/test
        w2v = Word2Vec(
            sentences=df_train["tokens"].tolist(),
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,          # 1 = skip-gram (mejor con palabras raras)
            workers=self.workers,
            epochs=10,
            seed=self.random_state,
        )
        print(f"Vocabulario word2vec: {len(w2v.wv):,} palabras")
        return w2v

    def _build_vectorizator(self):
        # Vocabulario = palabras que conoce word2vec.
        # TextVectorization reserva el índice 0 para padding y el 1 para [UNK].
        # Los tokens ya vienen separados por espacios, así que solo partimos por espacios
        # (así <url>, <happyface>, !! ... se mantienen como un único token).
        vocabulary = list(self.w2v.wv.index_to_key)

        return layers.TextVectorization(
            standardize=None,
            split="whitespace",
            output_mode="int",
            output_sequence_length=MAX_LEN,
            vocabulary=vocabulary,
        )

    def _build_embedding_matrix(self):
        final_vocabulary = self.vectorizator.get_vocabulary()  # ["", "[UNK]", "word1",...]
        embedding_matrix = np.zeros((len(final_vocabulary), self.vector_size), dtype="float32")

        for i, word in enumerate(final_vocabulary):
            if word in self.w2v.wv:
                embedding_matrix[i] = self.w2v.wv[word]

        # [UNK] recibe un vector aleatorio pequeño en lugar de ceros
        embedding_matrix[1] = np.random.normal(scale=0.1, size=self.vector_size)
        return embedding_matrix

    def _make_dataset(self, frame, shuffle=False):
        textos = frame["tokens"].str.join(" ").to_numpy()
        etiquetas = frame["label"].to_numpy(dtype="float32")
        ds = tf.data.Dataset.from_tensor_slices((textos, etiquetas))
        if shuffle:
            ds = ds.shuffle(buffer_size=100_000, seed=self.random_state, reshuffle_each_iteration=True)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.map(lambda x, y: (self.vectorizator(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def _build_model(self):
        vocab_size = self.embedding_matrix.shape[0]

        model = keras.Sequential([
            keras.Input(shape=(MAX_LEN,), dtype="int64"),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=self.vector_size,
                embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
                trainable=ENTRENAR_EMBEDDINGS,
                mask_zero=True,                     # ignoramos el padding
            ),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.4),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),  # probabilidad de "positivo"
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

        return model

    def fit(self, texts, labels):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"text": list(texts), "label": list(labels)})
        df["tokens"] = df["text"].apply(preprocess)

        df_train, df_val = train_test_split(df, test_size=VAL_SIZE, stratify=df["label"], random_state=self.random_state)

        if self.w2v_path.exists():
            self.w2v = Word2Vec.load(str(self.w2v_path))
        else:
            self.w2v = self._train_word2vec(df_train)
            self.w2v.save(str(self.w2v_path))

        self.vectorizator = self._build_vectorizator()

        if self.embeddings_path.exists():
            self.embedding_matrix = np.load(self.embeddings_path)
        else:
            self.embedding_matrix = self._build_embedding_matrix()
            np.save(self.embeddings_path, self.embedding_matrix)

        self.model = self._build_model()

        if self.weights_path.exists():
            self.model.load_weights(self.weights_path)
            return self

        ds_train = self._make_dataset(df_train, shuffle=True)
        ds_val = self._make_dataset(df_val)

        self.history = self.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=EPOCHS,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=2, restore_best_weights=True)],
        )

        self.model.save_weights(self.weights_path)

        return self

    def predict_proba(self, texts):
        tokens = [" ".join(preprocess(str(t))) for t in texts]
        ds = tf.data.Dataset.from_tensor_slices(tokens).batch(BATCH_SIZE)
        ds = ds.map(self.vectorizator, num_parallel_calls=tf.data.AUTOTUNE)
        return self.model.predict(ds, verbose=0).ravel()

    def predict(self, texts):
        return (self.predict_proba(texts) >= 0.5).astype(int)
