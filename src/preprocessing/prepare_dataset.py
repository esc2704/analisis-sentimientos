import io, zipfile, urllib.request
import re, html
import pandas as pd

from sklearn.model_selection import train_test_split

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

REAL_TRAIN = PROCESSED_DIR / "real_train.csv"
REAL_TEST = PROCESSED_DIR / "real_test.csv"
REAL_VAL = PROCESSED_DIR / "real_val.csv"


SYNTHETIC_PATHS = [
    PROCESSED_DIR / "synthetic_tweets.csv",
    RAW_DIR / "synthetic_tweets.csv",
]

RANDOM_SEED = 42

# Word2Vec
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
WORKERS = 4
SG = 1

_URL = "http://help.sentiment140.com/home"
_DATA_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

_TRAIN_FILE_NAME = "training.1600000.processed.noemoticon.csv"

SEED = 6

# Contracciones y abreviaciones tipicamente usadas en ingles
CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bain't\b", "is not"),
    (r"\b(it|that|what|there|he|she|who|here|where)'s\b", r"\1 is"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"), (r"'ll\b", " will"),
    (r"'ve\b", " have"), (r"'d\b", " would"), (r"'s\b", ""), (r"\bbtw\b", "by the way"),
    (r"\btbh\b", "to be honest"), (r"\bfr\b", "for real")
]
CONTRACTIONS = [(re.compile(p), r) for p, r in CONTRACTIONS]

# Corregimos apostrofes que puedan faltar, hoy en dia el corrector lo suele evitar
NO_APOSTROPHE = {
    "dont": "do not", "doesnt": "does not", "didnt": "did not", "isnt": "is not",
    "wasnt": "was not", "arent": "are not", "werent": "were not", "havent": "have not",
    "hasnt": "has not", "couldnt": "could not", "wouldnt": "would not",
    "shouldnt": "should not", "cant": "can not", "wont": "will not",
    "im": "i am", "ive": "i have", "youre": "you are", "thats": "that is",
}
NO_APOSTROPHE_RE = re.compile(r"\b(" + "|".join(NO_APOSTROPHE) + r")\b")

# Expresiones para detectar urls, usuarios, hastaghs, numeros, enfasis, :D, :(
URL_RE     = re.compile(r"https?://\S+|www\.\S+")
USER_RE    = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUM_RE     = re.compile(r"-?\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b")
LONG_RE  = re.compile(r"([a-z])\1{2,}")            # soooo -> soo
EXCLAM_RE  = re.compile(r"([!?])\1+")              # !!!!  -> !!
REST_RE   = re.compile(r"[^a-z<>!?\s]")            # solo letras, ! y ?
HAPPY_FACE = re.compile(r"[:;=]-?\)+|[:;=]-?d+\b") # ;) :) :d :))))
SAD_FACE   = re.compile(r"[:;=]-?\(+")             # :( :((( ;-(
TOKEN_RE   = re.compile(r"<url>|<user>|<hashtag>|<num>|<heart>|<happyface>|<sadface>|[a-z]+|[!?]{1,2}")


def preprocess(texto):
    t = html.unescape(texto).lower()                 # convertimos a minúsculas
    t = t.replace("’", "'").replace("`", "'")        # quitamos apostrofes raros
    t = URL_RE.sub(" <url> ", t)                     # marcamos urls
    t = USER_RE.sub(" <user> ", t)                   # marcamos usuarios
    t = HAPPY_FACE.sub(" <happyface> ", t)           # marcamos ;)
    t = SAD_FACE.sub(" <sadface> ", t)               # marcamos :(
    t = HASHTAG_RE.sub(r" <hashtag> \1 ", t)         # marcamos hastags
    t = t.replace("<3", " <heart> ")                 # marcamos corazones

    for patron, rep in CONTRACTIONS:
        t = patron.sub(rep, t)

    t = NO_APOSTROPHE_RE.sub(lambda m: NO_APOSTROPHE[m.group(1)], t)

    t = NUM_RE.sub(" <num> ", t)           # marcamos numeros
    t = LONG_RE.sub(r"\1\1", t)            # acortamos enfasis palabra
    t = EXCLAM_RE.sub(r" \1\1 ", t)        # acortamos enfasis exclamaciones
    t = REST_RE.sub(" ", t)                # eliminamos caracteres no permitidos

    return TOKEN_RE.findall(t)

print("hola")
def load_sentiment140():
    # Descargamos el zip con el Dataset
    with urllib.request.urlopen(_DATA_URL) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))

    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(zf.open(_TRAIN_FILE_NAME),
                     encoding="latin-1", header=None, names=cols)

    # Nos quedamos solo con target y text. Convertimos el sentimiento
    # que en el dataset es 0 = negativo, 4 = positivo a una c.binaria
    df = df[["target", "text"]].copy()
    df["label"] = (df["target"] == 4).astype(int)   # 0 = negativo, 1 = positivo

    return df


if __name__ == "__main__":
    df = load_sentiment140()

    # 0.85 train, 0.15 test
    df_train, df_test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_train[["text", "label"]].to_csv(REAL_TRAIN, index=False)
    df_test[["text", "label"]].to_csv(REAL_TEST, index=False)
