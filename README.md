# Uncertainty Estimation: comparación de métodos de incertidumbre en clasificación

Comparación reproducible de cinco métodos de estimación de incertidumbre en dos escenarios independientes de clasificación: **datos tabulares de riesgo crediticio** e **imágenes médicas**. El objetivo es medir qué tan bien calibrados están los modelos, si su incertidumbre permite **detectar errores** y cuánto baja el error cuando el modelo **se abstiene** en los casos más inciertos.

> **Estado: en desarrollo (semana 2).** Trabajo de Fin de Máster, Maestría en Inteligencia Artificial, curso 2026-27. Entorno reproducible, descarga de datos, EDA mínimo y métricas propias ya funcionando; los modelos y los métodos se añaden en las siguientes semanas (ver [Estado del proyecto](#estado-del-proyecto)).

## Resumen

Un clasificador entrega una probabilidad junto a su predicción, pero esa confianza no siempre refleja la realidad: muchos modelos dicen "99 % seguro" y aciertan mucho menos. En decisiones críticas (por ejemplo, riesgo crediticio o diagnóstico por imagen) es importante saber **cuándo un modelo no está seguro** para poder derivar el caso a una persona.

Este proyecto implementa y compara cinco métodos con el mismo protocolo experimental, las mismas particiones de datos y las mismas métricas, y publica el código para que cualquiera pueda reproducir los resultados o reutilizarlo en sus propios datos.

## Preguntas de investigación

1. ¿Qué método estima mejor la confianza de las predicciones (calibración)?
2. ¿La incertidumbre es mayor cuando el modelo se equivoca, de modo que sirva para detectar sus errores?
3. ¿Cuánto baja el error si el modelo se abstiene en los casos más inciertos y cuánto se pierde en cobertura?

## Qué se compara

| Método | Idea en una línea |
| --- | --- |
| Softmax (línea base) | Usa directamente la probabilidad máxima que entrega el modelo |
| *Temperature scaling* | Reescala los logits con un único parámetro ajustado en validación; corrige la sobreconfianza sin cambiar la clase predicha |
| MC Dropout | Mantiene el *dropout* activo al predecir y mide la dispersión entre varias pasadas |
| Deep Ensembles | Entrena varios modelos con semillas distintas y mide su desacuerdo |
| Predicción conforme (*split*) | Devuelve un conjunto de clases plausibles con garantía de cobertura bajo intercambiabilidad |

Modelos base sobre los que se aplican (los cinco métodos usan el mismo modelo): **MLP** para el escenario tabular (con regresión logística como línea base simple) y una **CNN pequeña** para imágenes, ambos con *dropout* antes del clasificador.

## Escenarios y datos

Los dos escenarios se evalúan **por separado**; no se fusionan datos tabulares con imágenes.

| Escenario | Conjunto de datos | Tarea | Tamaño verificado |
| --- | --- | --- | --- |
| Tabular | UCI *Default of Credit Card Clients* ([DOI](https://doi.org/10.24432/C55S3H)) | Clasificación binaria (impago) | 30 000 clientes, 23 variables, 22,12 % de impagos |
| Imágenes médicas | PneumoniaMNIST — MedMNIST v2 ([DOI](https://doi.org/10.1038/s41597-022-01721-8)) | Clasificación binaria (normal / neumonía) | 5 856 radiografías 28×28 (4 708 / 524 / 624) |

- **No se incluyen datos en el repositorio.** `scripts/download_data.py` los descarga desde la fuente oficial en un solo comando y escribe un manifiesto con URL, licencia, *checksum* y formas.
- Licencias: crédito **CC BY 4.0**; MedMNIST **CC BY 4.0** (el código de `medmnist` es Apache-2.0). Este repositorio solo distribuye código (MIT).
- Solo se usan conjuntos de datos públicos.

### Hallazgos del EDA mínimo (generados por script)

- **Crédito:** sin valores faltantes, pero **56 filas duplicadas** exactas en las variables (ignorando la etiqueta). Se eliminan **antes** de partir los datos para evitar fuga de información entre entrenamiento y prueba.
- **Crédito:** desbalance moderado (22 % de impagos) y variables sensibles (`sex`, `education`, `marriage`, `age`) con tasas de impago distintas, lo que se documenta como sesgo potencial.
- **PneumoniaMNIST:** la proporción de clase positiva **no es la misma** en entrenamiento/validación (0,742) que en el conjunto de prueba oficial (0,625). Hay que tenerlo en cuenta al comparar exactitud y al fijar umbrales.
- Informe completo y figuras: `results/eda/report.md` (regenerable con `scripts/run_eda.py`).

## Métricas

Implementadas **desde cero** en `src/uq/metrics/`, con pruebas de respuesta conocida:

- **Calibración:** ECE sobre la clase predicha (15 bins de ancho igual, con variante de igual masa), puntuación de Brier, NLL y curvas de fiabilidad.
- **Detección de errores:** AUROC y AUPRC de la puntuación de incertidumbre frente a "predicción errónea", más entropía predictiva en nats o bits.
- **Clasificación selectiva:** curva riesgo-cobertura, área bajo la curva, error a coberturas fijas y umbral correspondiente.
- **Predicción conforme:** cuantil con corrección de tamaño finito, conjuntos de predicción, cobertura empírica, tamaño medio del conjunto y cobertura por clase.
- **Variabilidad:** varias semillas por experimento, con media, desviación estándar e intervalos de confianza por *bootstrap* (pendiente).

## Protocolo experimental

- Particiones de entrenamiento, validación, calibración y prueba, estratificadas y con índices guardados en `data/splits/*.json`. El conjunto de prueba solo se usa en la evaluación final.
- **Crédito:** se excluyen las 56 filas duplicadas y se reparte 60/10/10/20 (17 966 / 2 994 / 2 994 / 5 990).
- **Imágenes:** el test oficial (624) queda intacto; los 5 232 de train+val se reparten 80/10/10 (4 185 / 523 / 524).
- El umbral de abstención y el parámetro de temperatura se ajustan con validación; la predicción conforme usa un conjunto de calibración separado.
- Mismos modelos base, mismas particiones y mismo presupuesto de ajuste para todos los métodos.
- Semillas fijas y registradas; versiones de las dependencias bloqueadas en un archivo de bloqueo.
- Todas las tablas y figuras se generan por script desde los resultados guardados.

## Requisitos

| Requisito | Detalle |
| --- | --- |
| Python | **3.13 o superior**. Desarrollado y probado con **3.14.10** (ver [decisiones técnicas](#decisiones-técnicas-y-por-qué)). |
| Gestor de entorno | [`uv`](https://docs.astral.sh/uv/) — crea el entorno y respeta `uv.lock`. |
| Sistema operativo | Windows, Linux o macOS. |
| GPU | **Opcional.** Sin GPU todo funciona (más lento en imágenes). Probado con NVIDIA RTX 3090 (24 GB) y CUDA 13.2. |
| Espacio en disco | ~5 GB (entorno con PyTorch CUDA) + ~50 MB de datos. |

## Instalación

```bash
# 1. Instalar uv (una sola vez)
#    Windows
winget install --id astral-sh.uv -e
#    macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clonar el repositorio
git clone https://github.com/08edu02/uncertainty-estimation-tfm.git
cd uncertainty-estimation-tfm

# 3. Crear el entorno e instalar las versiones bloqueadas
uv sync

# 4. Comprobar que todo está en su sitio
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

En Windows y Linux, `uv sync` instala PyTorch desde el índice oficial de CUDA 13.2; en macOS instala la build estándar (MPS). Si el equipo no tiene GPU, el mismo entorno funciona en CPU: `torch.cuda.is_available()` devuelve `False` y no hay que cambiar nada más.

## Uso

```bash
# Descargar los datos desde la fuente oficial (una vez; escribe data/manifest.json)
uv run python scripts/download_data.py

# EDA mínimo: results/eda/*.json, results/eda/report.md y figuras
uv run python scripts/run_eda.py

# Calidad de código y pruebas
uv run ruff check .
uv run ruff format --check .
uv run pytest

# Exploración interactiva (abrir en VS Code / Jupyter con el kernel del .venv)
#   notebooks/01_eda_credit.ipynb
#   notebooks/02_eda_pneumonia.ipynb
```

Variables de entorno opcionales: `UQ_DATA_DIR`, `UQ_RESULTS_DIR`, `UQ_FIGURES_DIR` (por defecto `./data`, `./results` y `./results/figures`).

> Los scripts de entrenamiento, evaluación y de generación de tablas/figuras de la tesis (`scripts/run_all.py`) se añaden en las próximas semanas.

## Estructura del repositorio

```text
uncertainty-estimation-tfm/
├── pyproject.toml            # dependencias, lint, pytest (PEP 621)
├── uv.lock                   # versiones exactas bloqueadas
├── .python-version           # 3.14
├── CITATION.cff
├── src/uq/
│   ├── settings.py           # rutas y variables de entorno
│   ├── data/                 # credit.py, pneumonia.py, splits.py
│   ├── models/               # mlp.py (tabular), cnn.py (28×28)
│   ├── eda/                  # summary.py (JSON), figures.py (PNG)
│   └── metrics/              # calibración, detección de errores, selectiva, conforme
├── scripts/
│   ├── download_data.py
│   └── run_eda.py
├── notebooks/                # EDA y exploración (sin lógica crítica)
├── tests/                    # carga de datos, particiones, métricas, modelos y rutas
├── results/
│   ├── eda/                  # resúmenes JSON + informe (versionados)
│   └── figures/              # figuras (regenerables, ignoradas por git)
└── .github/workflows/ci.yml  # lint + pruebas + lockfile
```

Pendiente de las próximas semanas: `configs/`, `src/uq/methods/`, `src/uq/train.py`, `src/uq/evaluate.py` y `scripts/run_all.py`.

El código y los nombres de variables están en inglés (para que el repositorio sea reutilizable fuera del ámbito hispanohablante); la documentación de usuario, los informes generados y la tesis están en español.

## Versiones

Verificadas en las fuentes oficiales (PyPI y python.org) el **18 de noviembre de 2026**. `uv.lock` es la fuente de verdad: fija las versiones exactas que se usan en cada resultado.

| Componente | Versión | Nota |
| --- | --- | --- |
| Python | 3.14.10 | 3.15.2 es la última estable, pero `scikit-image` (dependencia de `medmnist`) aún no publica *wheels* para ella |
| PyTorch | 2.15.0+cu132 | Instalado desde el índice oficial CUDA 13.2 |
| torchvision | 0.29.2 | |
| NumPy | 2.5.5 | |
| pandas | 3.1.1 | |
| scikit-learn | 1.9.1 | |
| SciPy | 1.18.1 | |
| Matplotlib | 3.11.2 | |
| seaborn | 0.13.2 | |
| medmnist | 3.0.2 | |
| pytest | 9.2.0 | |
| Ruff | 0.17.5 | |
| uv | 0.13.9 | |

## Decisiones técnicas y por qué

- **Python 3.14 y no 3.15.** 3.15.2 es la última versión estable, pero al instalar el entorno `scikit-image` no tenía *wheel* para 3.15 y la compilación desde fuente fallaba. Se prefiere la versión más reciente que **todo el ecosistema soporta** antes que forzar compilaciones. Actualizar a 3.15 es una línea en `.python-version` cuando haya *wheels*.
- **PyTorch desde el índice CUDA 13.2.** En Windows, el `torch` que publica PyPI es la build de CPU (127 MB): no habilita la GPU. El índice oficial `pytorch-cu132` sí ofrece la build CUDA para esta versión de Python y es el canal más reciente con *wheels* publicados.
- **`engine="calamine"` para leer el Excel del UCI.** El archivo original es un `.xls` antiguo; en lugar de añadir `xlrd`, se usa `python-calamine`, el lector que pandas 3 recomienda para Excel.
- **Nada de lógica en los notebooks.** Los notebooks llaman a `uq.*`; los informes y figuras los generan los scripts, de modo que toda cifra de la tesis se regenera con un comando.

## Resultados

*Pendiente.* Esta sección mostrará las tablas comparativas y las figuras principales (diagramas de fiabilidad, curvas riesgo-cobertura y resultados de la predicción conforme) cuando terminen los experimentos.

## Limitaciones y uso responsable

- Los conjuntos de datos son públicos y reducidos. **Los resultados son de investigación: no sustituyen una validación clínica ni deben usarse para decisiones reales de crédito.**
- El conjunto de crédito contiene variables sensibles (como sexo, edad o estado civil), que pueden introducir sesgos. No se extraen conclusiones sociales de este trabajo.
- El estudio se limita a clasificación y no cubre datos fuera de distribución, regresión ni modelos de lenguaje.
- La predicción conforme ofrece una garantía **marginal** (en promedio sobre los datos), no condicional por clase o subgrupo.

## Estado del proyecto

- [x] Estructura del repositorio y entorno reproducible (Python 3.14, `uv`, CUDA verificado)
- [x] Análisis exploratorio mínimo de los dos conjuntos de datos
- [x] Métricas implementadas y probadas
- [ ] Líneas base y *temperature scaling*
- [ ] MC Dropout y Deep Ensembles
- [ ] Predicción conforme
- [ ] Experimentos completos en los dos escenarios
- [ ] Tablas y figuras generadas por script
- [ ] Documentación final y versión etiquetada del trabajo entregado

## Hoja de ruta

- **Enero de 2027:** metodología validada y primeros resultados.
- **Finales de enero de 2027:** experimentos completos y resultados.
- **Marzo de 2027:** versión etiquetada del trabajo entregado y documentación final.

Las fechas pueden variar según el calendario del trabajo.

## Cómo citar

Los metadatos están en [`CITATION.cff`](CITATION.cff). Mientras el trabajo no esté terminado:

```bibtex
@misc{uncertainty_estimation_tfm,
  author = {Carlos Eduardo Chavarria Mujica},
  title  = {Uncertainty Estimation: comparación de métodos de incertidumbre en clasificación},
  year   = {2027},
  note   = {Trabajo de Fin de Máster, VIU},
  url    = {https://github.com/08edu02/uncertainty-estimation-tfm}
}
```

## Licencia

Código bajo licencia **MIT**. Los conjuntos de datos tienen sus propias licencias (CC BY 4.0) y no forman parte de este repositorio.

## Referencias principales

<!-- TODO: doble check de los artículos, fechas y que guarden relación con el trabajo -->

- Guo, Pleiss, Sun y Weinberger (2017). *On Calibration of Modern Neural Networks*. ICML.
- Gal y Ghahramani (2016). *Dropout as a Bayesian Approximation*. ICML.
- Lakshminarayanan, Pritzel y Blundell (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS.
- Angelopoulos y Bates (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*.
- Geifman y El-Yaniv (2017). *Selective Classification for Deep Neural Networks*. NeurIPS.
- Naeini, Cooper y Hauskrecht (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning*. AAAI.
- Yang et al. (2023). *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific Data.
- Yeh y Lien (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications.

## Agradecimientos

<!-- TODO: asesor, universidad y otras personas o instituciones -->
