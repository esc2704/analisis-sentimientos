import io, zipfile, urllib.request
import re, html, sys
import pandas as pd

from sklearn.model_selection import train_test_split

from pathlib import Path

# Añadimos la raiz del proyecto al path para poder importar config tanto con
# `python preprocessing/prepare_dataset.py` como con `python -m preprocessing.prepare_dataset`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *

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

def download_zip(url):
    # Descargamos el zip en memoria
    print(f"Descargando {url} ...")
    with urllib.request.urlopen(url) as r:
        return zipfile.ZipFile(io.BytesIO(r.read()))


def clean(df):
    # Quitamos filas vacias y tuits duplicados (mismo texto)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)

    labels = set(df["label"].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Etiquetas no validas: {labels}")

    return df[["text", "label"]].reset_index(drop=True)


def load_sentiment140():
    zf = download_zip(SEN_DATA_URL)

    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(zf.open(SEN_TRAIN_FILE_NAME),
                     encoding="latin-1", header=None, names=cols)

    # Nos quedamos solo con target y text. Convertimos el sentimiento
    # que en el dataset es 0 = negativo, 4 = positivo a una c.binaria
    df = df[["target", "text"]].copy()
    df["label"] = (df["target"] == 4).astype(int)   # 0 = negativo, 1 = positivo

    return clean(df)


def load_synthetic():
    zf = download_zip(SYN_DATA_URL)

    df = pd.read_csv(zf.open(SYN_TRAIN_FILE_NAME), encoding="utf-8")

    # Nos quedamos solo con text y label (ya viene como 0 = negativo, 1 = positivo)
    df = df[["text", "label"]].copy()

    return clean(df)


def split(df):
    # Primero separamos test y despues dividimos el resto en train y val,
    # estratificando por la etiqueta para mantener la proporcion de clases
    df_rest, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED)

    val_rel = VAL_SIZE / (1 - TEST_SIZE)
    df_train, df_val = train_test_split(
        df_rest, test_size=val_rel, stratify=df_rest["label"], random_state=RANDOM_SEED)

    return df_train, df_val, df_test


def save_splits(name, df, train_path, val_path, test_path):
    df_train, df_val, df_test = split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for part, path in [(df_train, train_path), (df_val, val_path), (df_test, test_path)]:
        part.to_csv(path, index=False)

    print(f"\n{name}: {len(df)} tuits")
    for split_name, part, path in [("train", df_train, train_path),
                                   ("val", df_val, val_path),
                                   ("test", df_test, test_path)]:
        pos = part["label"].mean()
        print(f"  {split_name:<5} {len(part):>8}  ({pos:.1%} positivos) -> {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    save_splits("Sentiment140", load_sentiment140(), REAL_TRAIN, REAL_VAL, REAL_TEST)
    save_splits("Sintetico", load_synthetic(), SYN_TRAIN, SYN_VAL, SYN_TEST)
