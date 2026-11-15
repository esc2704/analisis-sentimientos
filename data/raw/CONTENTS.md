# RAW

En esta carpeta se muestra un trozo del dataset sentiment140 publicado por Standford. Dejamos el link de HuggingFace para descargar el dataset junto con la ficha del conjunto de datos.

https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

# Ficha del conjunto de datos «sentiment140»

Sentiment140 es un conjunto de datos de gran escala para la clasificación del sentimiento en Twitter, que contiene 1,6 millones de tuits etiquetados automáticamente mediante emoticonos como forma de supervisión distante (*distant supervision*). Los tuits que contienen emoticonos positivos (p. ej., :), :-)) se etiquetan como positivos (`sentiment=4`), mientras que los que contienen emoticonos negativos (p. ej., :(, :-() se etiquetan como negativos (`sentiment=0`). Este conjunto de datos se utiliza ampliamente como referencia para la clasificación de sentimientos binaria y multiclase en la investigación sobre procesamiento del lenguaje natural (PLN).

## Tareas compatibles y clasificaciones de referencia

**Clasificación de texto / Análisis de sentimientos:** El conjunto de datos está diseñado para la clasificación binaria de sentimientos (negativo frente a positivo). El campo `sentiment` se utiliza como etiqueta. Los modelos suelen evaluarse utilizando la partición de prueba de 498 ejemplos, mediante las métricas de exactitud (*accuracy*) y puntuación F1 (*F1 score*).

Los resultados de las clasificaciones y las evaluaciones de referencia pueden consultarse en Papers With Code — Sentiment140.

## Idiomas

El conjunto de datos contiene tuits en inglés (`en`), recopilados de la plataforma Twitter.

## Estructura del conjunto de datos

### Instancias de datos

**sentiment140**

- Tamaño de los archivos del conjunto de datos descargados: 81,36 MB.
- Tamaño del conjunto de datos generado: 225,82 MB.
- Espacio total en disco utilizado: 307,18 MB.

Un ejemplo de la partición `train` tiene la siguiente estructura:

    {
        "date": "23-04-2010",
        "query": "NO_QUERY",
        "sentiment": 3,
        "text": "train message",
        "user": "train user"
    }

### Campos de datos

Los campos de datos son los mismos en todas las particiones.

**sentiment140**

- **text**: característica de tipo cadena de caracteres. Contiene el texto original del tuit.
- **date**: característica de tipo cadena de caracteres. Indica la fecha y hora en que se publicó el tuit.
- **user**: característica de tipo cadena de caracteres. Nombre de usuario de Twitter del autor.
- **sentiment**: característica de tipo `int32`. Etiqueta de sentimiento: 0 = negativo, 2 = neutro, 4 = positivo.
- **query**: característica de tipo cadena de caracteres. Palabra clave utilizada para recuperar el tuit o `NO_QUERY` si no se utilizó ninguna.

### División de los datos

| Nombre | Entrenamiento (train) | Prueba (test) |
|---|---:|---:|
| sentiment140 | 1.600.000 | 498 |

## Creación del conjunto de datos

### Justificación de la recopilación

El conjunto de datos se creó para abordar la falta de datos etiquetados a gran escala para el análisis de sentimientos en Twitter. En lugar de recurrir a la anotación manual, los autores utilizaron un enfoque de supervisión distante (*distant supervision*): los emoticonos presentes en los tuits sirven como etiquetas de sentimiento, aunque sean ruidosas, pero permiten recopilar datos a gran escala.

Este enfoque permitió reunir 1,6 millones de ejemplos etiquetados sin necesidad de recurrir a anotadores humanos.

## Datos de origen

### Recopilación inicial y normalización de los datos

Los tuits se recopilaron mediante la API de Twitter, realizando búsquedas de publicaciones que contenían emoticonos positivos o negativos. Se descartaron los tuits que contenían tanto emoticonos positivos como negativos.

Los nombres de usuario y las URL se sustituyeron por marcadores de posición uniformes (`@user`, `URL`) para reducir el ruido de los datos.

El conjunto de datos resultante se dividió en un conjunto de entrenamiento de gran tamaño (1,6 millones de tuits) y un conjunto de prueba etiquetado manualmente (498 tuits).

### ¿Quiénes son los productores del texto original?

El texto fue generado por usuarios habituales de Twitter que escribían en inglés.

El conjunto de datos fue recopilado y procesado por investigadores de la Universidad de Stanford: Alec Go, Richa Bhayani y Lei Huang, como parte de un proyecto de la asignatura CS224N.

## Anotaciones

### Proceso de anotación

Las etiquetas del conjunto de entrenamiento se asignaron automáticamente utilizando emoticonos como indicadores del sentimiento (*distant supervision*). No se utilizó anotación humana para el conjunto de entrenamiento.

El conjunto de prueba, compuesto por 498 tuits, fue etiquetado manualmente por los autores con fines de evaluación.

### ¿Quiénes son los anotadores?

- **Conjunto de entrenamiento:** no participaron anotadores humanos; las etiquetas se generaron automáticamente a partir de los emoticonos.
- **Conjunto de prueba:** los 498 tuits fueron etiquetados manualmente por los autores del estudio (Go, Bhayani y Huang) en la Universidad de Stanford.

## Información personal y sensible

El conjunto de datos contiene nombres de usuario reales de Twitter y contenido de tuits publicados desde cuentas públicas.

Aunque los datos estaban disponibles públicamente en el momento de su recopilación, es posible que los usuarios no fueran conscientes de que sus publicaciones se utilizarían con fines de investigación.

Los investigadores que utilicen este conjunto de datos deben tener presentes las posibles implicaciones relacionadas con la privacidad, especialmente al trabajar con el campo `user`.

## Consideraciones sobre el uso de los datos

### Impacto social del conjunto de datos

Sentiment140 se ha utilizado ampliamente para entrenar y evaluar modelos de procesamiento del lenguaje natural (PLN) destinados al análisis de sentimientos en redes sociales.

Los modelos entrenados con estos datos tienen aplicaciones en ámbitos como la monitorización de marcas, la investigación de la opinión pública y la detección de situaciones de crisis.

Sin embargo, dado que las etiquetas se obtienen a partir de emoticonos, el conjunto de datos puede no representar adecuadamente sentimientos matizados o ambiguos, lo que podría afectar a la fiabilidad de los modelos desarrollados posteriormente.

### Análisis de los posibles sesgos

- **Sesgo relacionado con los emoticonos:** las etiquetas se obtienen a partir de emoticonos, cuyo uso es más habitual en determinados grupos demográficos. Esto puede provocar una representación insuficiente de los usuarios que expresan sus sentimientos únicamente mediante el lenguaje escrito.

- **Sesgo temático:** los tuits se recopilaron mediante búsquedas por palabras clave, por lo que determinados temas y acontecimientos pueden estar sobrerrepresentados.

- **Sesgo temporal:** los tuits se recopilaron entre 2009 y 2010. Desde entonces, el uso del lenguaje, la jerga y los temas tratados en Twitter han evolucionado considerablemente.

- **Desequilibrio de clases en el conjunto de prueba:** el conjunto de prueba contiene únicamente 498 ejemplos, por lo que puede no ser representativo de las distribuciones de datos del mundo real.

### Otras limitaciones conocidas

- El conjunto de entrenamiento no contiene tuits neutros, sino únicamente positivos y negativos, etiquetados mediante emoticonos. La clase `sentiment=2` (neutro) solo aparece en el conjunto de prueba.
- El texto de los tuits puede contener ruido, como errores ortográficos, abreviaturas y formatos específicos de la plataforma.
- El conjunto de datos procede del periodo 2009-2010, por lo que puede no generalizar adecuadamente a textos actuales de redes sociales.

## Información adicional

### Responsables de la creación del conjunto de datos

El conjunto de datos fue creado por Alec Go, Richa Bhayani y Lei Huang, de la Universidad de Stanford, como parte de la asignatura de Procesamiento del Lenguaje Natural CS224N.

Está alojado en Hugging Face Hub por los usuarios `@patrickvonplaten` y `@thomwolf`.

### Información sobre la licencia

Los autores pusieron el conjunto de datos a disposición del público de forma gratuita para fines de investigación. No se ha asociado ninguna licencia explícita de código abierto al conjunto de datos.

Los usuarios deben consultar la política para desarrolladores de Twitter (*Twitter Developer Policy*) en relación con el uso del contenido de los tuits.

### Información para citar el conjunto de datos

    @article{go2009twitter,
      title={Twitter sentiment classification using distant supervision},
      author={Go, Alec and Bhayani, Richa and Huang, Lei},
      journal={CS224N project report, Stanford},
      volume={1},
      number={12},
      pages={2009},
      year={2009}
    }

## Contribuciones

Agradecimientos a `@patrickvonplaten` y `@thomwolf` por incorporar este conjunto de datos.