import numpy as np

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression


class Word2VecClassifier:

    def __init__(
        self,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        random_state=42
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.random_state = random_state

        self.word2vec = None
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )

    def fit(self, texts, labels):

        sentences = [
            str(text).split()
            for text in texts
        ]

        self.word2vec = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            seed=self.random_state
        )

        embeddings = self.transform(texts)

        self.classifier.fit(embeddings, labels)

        return self

    def transform(self, texts):

        embeddings = []

        for text in texts:

            words = str(text).split()

            vectors = [
                self.word2vec.wv[word]
                for word in words
                if word in self.word2vec.wv
            ]

            if vectors:
                tweet_vector = np.mean(vectors, axis=0)
            else:
                tweet_vector = np.zeros(self.vector_size)

            embeddings.append(tweet_vector)

        return np.array(embeddings)

    def predict(self, texts):

        embeddings = self.transform(texts)

        return self.classifier.predict(embeddings)