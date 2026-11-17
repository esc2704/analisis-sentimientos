# analisis-sentimientos
Análisis de sentimientos mediante técnicas de procesamiento del lenguaje natural, utilizando el conjunto de datos Sentiment140 de Stanford y un database generado con IA (solo en train), comparando bag Of Words y Word2Vec, empleando para ambos clasificación mediante Regresión Logística.


# ejecutar

## instalar requirements
python3 -m venv venv_analisis_sentimiento
source venv_analisis_sentimiento/bin/activate
pip install -r requirements.txt

## ejecuta experiments y compare_models
rm -f results/metrics/*.csv
python -m experiments.run_all

# Flujo de ejecución del proyecto

## 1. Preparación de los datos

```mermaid
flowchart TD
    A["dataset Sentiment140 original"] --> B["src/preprocessing/<br/> prepare_dataset.py"]
    B --> C["data/processed/"]
    C --> D["real_train.csv"]
    C --> E["real_val.csv"]
    C --> F["real_test.csv"]

    I["dataset LLM"] --> J["src/synthetic/<br/>generate_tweets.py"]
    J --> K["data/raw/<br/>synthetic_tweets.csv"]
```

## 2. Como funciona la ejecución de los experimentos

```mermaid
flowchart TD
    A["python -m experiments.run_all"] --> B["experiments/run_all.py"]
    B --> C["experiments/common.py"]

    C --> D["Carga de datos<br/>real_train.csv"]
    C --> E["Carga de datos<br/>synthetic_tweets.csv"]
    C --> F["Carga de datos<br/>real_test.csv"]

    D --> G["Selección del escenario"]
    E --> G

    G --> H["REAL"]
    G --> I["SYNTHETIC"]
    G --> J["COMBINED"]

    H --> K["BOW"]
    H --> L["Word2Vec"]
    I --> M["BOW"]
    I --> N["Word2Vec"]
    J --> O["BOW"]
    J --> P["Word2Vec"]

    K --> Q["evaluation/metrics.py"]
    L --> Q
    M --> Q
    N --> Q
    O --> Q
    P --> Q

    F --> Q
```

## 3. Cuales son los modelos utilizados

```mermaid
flowchart TD
    A["Selección del modelo"]

    A --> B["BOW"]
    A --> C["Word2Vec"]

    B --> F["Clasificador Regresión Logística"]

    C --> D["Promedio de embeddings"]
    D --> E["Clasificador Regresión Logística"]

    E --> G["Predicciones"]
    F --> G
```

## 4. Evaluación y resultados

```mermaid
flowchart TD
    A["Predicciones"] --> B["evaluation/metrics.py"]

    B --> D["Accuracy"]
    B --> E["Precision"]
    B --> F["Recall"]
    B --> G["F1-score"]

    B --> H["Matrices de confusión"]
    B --> I["Predicciones por tweet"]

    D --> J["results/metrics/"]
    E --> J
    F --> J
    G --> J

    H --> K["results/figures/"]
    I --> L["results/predictions/"]

    J --> M["evaluation/compare_models.py"]
    M --> N["comparison.csv"]
```

## 5. Estructura de resultados

```text
results/
│
├── metrics/
│   ├── bow_real.csv
│   ├── bow_synthetic.csv
│   ├── bow_combined.csv
│   ├── word2vec_real.csv
│   ├── word2vec_synthetic.csv
│   ├── word2vec_combined.csv
│   └── comparison.csv
│
├── figures/
│   ├── bow_real_confusion.png
│   ├── bow_synthetic_confusion.png
│   ├── bow_combined_confusion.png
│   ├── word2vec_real_confusion.png
│   ├── word2vec_synthetic_confusion.png
│   └── word2vec_combined_confusion.png
│
└── predictions/
    ├── bow_real_predictions.csv
    ├── bow_synthetic_predictions.csv
    ├── bow_combined_predictions.csv
    ├── word2vec_real_predictions.csv
    ├── word2vec_synthetic_predictions.csv
    └── word2vec_combined_predictions.csv
```