import re
import html
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from config import REAL_TRAIN, REAL_VAL, REAL_TEST
from evaluation.metrics import evaluate_model


# --------------------------------------------------
# PREPROCESAMIENTO DE TEXTO
# --------------------------------------------------

CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bain't\b", "is not"),
    (r"\b(it|that|what|there|he|she|who|here|where)'s\b", r"\1 is"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"), (r"'ll\b", " will"),
    (r"'ve\b", " have"), (r"'d\b", " would"), (r"'s\b", ""), (r"\bbtw\b", "by the way"),
    (r"\btbh\b", "to be honest"), (r"\bfr\b", "for real")
]
CONTRACTIONS = [(re.compile(p), r) for p, r in CONTRACTIONS]

NO_APOSTROPHE = {
    "dont": "do not", "doesnt": "does not", "didnt": "did not", "isnt": "is not",
    "wasnt": "was not", "arent": "are not", "werent": "were not", "havent": "have not",
    "hasnt": "has not", "couldnt": "could not", "wouldnt": "would not",
    "shouldnt": "should not", "cant": "can not", "wont": "will not",
    "im": "i am", "ive": "i have", "youre": "you are", "thats": "that is",
}
NO_APOSTROPHE_RE = re.compile(r"\b(" + "|".join(NO_APOSTROPHE) + r")\b")

URL_RE     = re.compile(r"https?://\S+|www\.\S+")
USER_RE    = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUM_RE     = re.compile(r"-?\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b")
LONG_RE    = re.compile(r"([a-z])\1{2,}")            # soooo -> soo
EXCLAM_RE  = re.compile(r"([!?])\1+")              # !!!!  -> !!
REST_RE    = re.compile(r"[^a-z<>!?\s]")            # solo letras, ! y ?
HAPPY_FACE = re.compile(r"[:;=]-?\)+|[:;=]-?d+\b") # ;) :) :d :))))
SAD_FACE   = re.compile(r"[:;=]-?\(+")             # :( :((( ;-(
TOKEN_RE   = re.compile(r"<url>|<user>|<hashtag>|<num>|<heart>|<happyface>|<sadface>|[a-z]+|[!?]{1,2}")


def preprocess(texto):
    t = html.unescape(str(texto)).lower()            # convertimos a minúsculas
    t = t.replace("\u2019", "'").replace("`", "'")   # quitamos apóstrofos raros
    t = URL_RE.sub(" <url> ", t)                     # marcamos urls
    t = USER_RE.sub(" <user> ", t)                   # marcamos usuarios
    t = HAPPY_FACE.sub(" <happyface> ", t)           # marcamos ;)
    t = SAD_FACE.sub(" <sadface> ", t)               # marcamos :(
    t = HASHTAG_RE.sub(r" <hashtag> \1 ", t)         # marcamos hashtags
    t = t.replace("<3", " <heart> ")                 # marcamos corazones

    for patron, rep in CONTRACTIONS:
        t = patron.sub(rep, t)

    t = NO_APOSTROPHE_RE.sub(lambda m: NO_APOSTROPHE[m.group(1)], t)

    t = NUM_RE.sub(" <num> ", t)           # marcamos numeros
    t = LONG_RE.sub(r"\1\1", t)            # acortamos enfasis palabra
    t = EXCLAM_RE.sub(r" \1\1 ", t)        # acortamos enfasis exclamaciones
    t = REST_RE.sub(" ", t)                # eliminamos caracteres no permitidos

    return TOKEN_RE.findall(t)


def preprocess_text(texto):
    return " ".join(preprocess(texto))


# --------------------------------------------------
# CREACIÓN DEL MODELO BOW
# --------------------------------------------------

def create_bow():
    # BOW -> Regresión logística
    return Pipeline([
        (
            "vectorizador",
            CountVectorizer(
                lowercase=True,
                token_pattern=r"<url>|<user>|<hashtag>|<num>|<heart>|<happyface>|<sadface>|[a-z]+|[!?]{1,2}"
            )
        ),
        (
            "modelo",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])


# --------------------------------------------------
# CARGA DE LOS DATOS
# --------------------------------------------------

def load_dataset(path):
    df = pd.read_csv(path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            f"El archivo {path} debe contener text y label."
        )

    df = df[["text", "label"]].dropna().copy()

    df["text"] = df["text"].astype(str).apply(preprocess_text)
    df["label"] = pd.to_numeric(df["label"])

    df["label"] = df["label"].replace({4: 1})

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Las etiquetas deben ser 0 o 1.")

    df["label"] = df["label"].astype(int)

    return df


# --------------------------------------------------
# ENTRENAMIENTO Y EVALUACIÓN
# --------------------------------------------------

def run_bow():
    # Cargar los conjuntos procesados
    train = load_dataset(REAL_TRAIN)
    val = load_dataset(REAL_VAL)
    test = load_dataset(REAL_TEST)

    textos_train = train["text"]
    etiquetas_train = train["label"]

    textos_val = val["text"]
    etiquetas_val = val["label"]

    textos_test = test["text"]
    etiquetas_test = test["label"]

    print("Ejemplos de entrenamiento:", len(train))
    print("Ejemplos de validación:", len(val))
    print("Ejemplos de test:", len(test))

    # Crear el modelo
    modelo = create_bow()

    # Entrenar con train
    modelo.fit(textos_train, etiquetas_train)

    # Predecir validación
    predicciones_val = modelo.predict(textos_val)

    # Evaluar validación
    evaluate_model(
        etiquetas_val,
        predicciones_val,
        "bow",
        "real_val"
    )

    # Reentrenar con train + validation
    textos_train_final = pd.concat(
        [textos_train, textos_val]
    )

    etiquetas_train_final = pd.concat(
        [etiquetas_train, etiquetas_val]
    )

    modelo.fit(
        textos_train_final,
        etiquetas_train_final
    )

    # Predecir test
    predicciones_test = modelo.predict(textos_test)

    # Evaluar test
    evaluate_model(
        etiquetas_test,
        predicciones_test,
        "bow",
        "real"
    )

    return modelo, predicciones_val, predicciones_test


# --------------------------------------------------
# VOCABULARIO, MATRIZ BOW Y COEFICIENTES
# --------------------------------------------------

def inspect_bow(modelo, textos):
    vectorizador = modelo.named_steps["vectorizador"]
    clasificador = modelo.named_steps["modelo"]

    vocabulary = vectorizador.get_feature_names_out()
    bow_matrix = vectorizador.transform(textos)
    coefficients = clasificador.coef_[0]

    resultados = pd.DataFrame({
        "palabra": vocabulary,
        "coeficiente": coefficients
    })

    positivas = resultados.sort_values(
        by="coeficiente",
        ascending=False
    )

    negativas = resultados.sort_values(
        by="coeficiente",
        ascending=True
    )

    print("\nPalabras asociadas a sentimiento positivo:")
    print(positivas.head(20))

    print("\nPalabras asociadas a sentimiento negativo:")
    print(negativas.head(20))

    print("\nDimensiones de la matriz BOW:")
    print(bow_matrix.shape)

    return vocabulary, bow_matrix, coefficients


# --------------------------------------------------
# PREDICCIÓN SOBRE FRASES NUEVAS
# --------------------------------------------------

def predecir(modelo, textos):
    tokens_procesados = [preprocess_text(t) for t in textos]
    probs = modelo.predict_proba(tokens_procesados)[:, 1]
    preds = modelo.predict(tokens_procesados)

    print("\nPredicciones sobre frases nuevas:")
    for texto, p, pred in zip(textos, probs, preds):
        etiqueta = "positivo" if pred == 1 else "negativo"
        print(f"{p:.3f}  {etiqueta:<8}  {texto}")


# --------------------------------------------------
# EJECUCIÓN
# --------------------------------------------------

if __name__ == "__main__":
    modelo, predicciones_val, predicciones_test = run_bow()

    train = load_dataset(REAL_TRAIN)
    val = load_dataset(REAL_VAL)

    textos_train_final = pd.concat([
        train["text"],
        val["text"]
    ])

    vocabulary, bow_matrix, coefficients = inspect_bow(
        modelo,
        textos_train_final
    )

    predecir(modelo, [
        "I love love movies",
        "I can't believe how bad this day was :(",
        "just got my new phone, sooo happy!!!",
        "I don't hate it, it's actually not bad",
        "worst customer service ever, never again",
        "@john thanks for the help yesterday <3",
        "hello",
        "bye",
        ":(",
        ":)",
        ":) :(",
        ":( :)",
        "<3",
        "fuck",
        "heheh"
    ])