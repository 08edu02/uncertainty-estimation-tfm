# Uncertainty Estimation: comparación de métodos de incertidumbre en clasificación

Comparación reproducible de cinco métodos de estimación de incertidumbre en dos escenarios independientes de clasificación: **datos tabulares de riesgo crediticio** e **imágenes médicas**. El objetivo es medir qué tan bien calibrados están los modelos, si su incertidumbre permite **detectar errores** y cuánto baja el error cuando el modelo **se abstiene** en los casos más inciertos.

> **Estado: en desarrollo.** Este repositorio acompaña un Trabajo de Fin de Máster (Maestría en Inteligencia Artificial, curso 2026-27). Las secciones marcadas como *planificado* describen lo que habrá al terminar el trabajo; la sección de estado indica qué está listo en cada momento.

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

## Escenarios y datos

Los dos escenarios se evalúan **por separado**; no se fusionan datos tabulares con imágenes.

| Escenario | Conjunto de datos (candidato) | Tarea |
| --- | --- | --- |
| Tabular | UCI "Default of Credit Card Clients" | Clasificación binaria (impago) |
| Imágenes médicas | MedMNIST (por ejemplo, PneumoniaMNIST) | Clasificación de imágenes |

- **No se incluyen datos en el repositorio.** Un script los descargará desde la fuente oficial. Consulta la licencia y las condiciones de uso de cada conjunto antes de reutilizarlos.
- Solo se usan conjuntos de datos públicos.
- Los conjuntos concretos pueden ajustarse tras el análisis exploratorio inicial.

## Métricas

- **Calibración:** ECE, puntuación de Brier, NLL y diagramas de fiabilidad.
- **Detección de errores:** AUROC de la puntuación de incertidumbre frente a "predicción errónea".
- **Clasificación selectiva:** curva riesgo-cobertura, área bajo la curva y error a coberturas fijas.
- **Predicción conforme:** cobertura empírica, tamaño medio del conjunto y cobertura por clase.
- **Variabilidad:** varias semillas por experimento, con media, desviación estándar e intervalos de confianza por *bootstrap*.

## Protocolo experimental

- Particiones de entrenamiento, validación, calibración y prueba, estratificadas y con índices guardados. El conjunto de prueba solo se usa en la evaluación final.
- El umbral de abstención y el parámetro de temperatura se ajustan con validación; la predicción conforme usa un conjunto de calibración separado.
- Mismos modelos base, mismas particiones y mismo presupuesto de ajuste para todos los métodos.
- Semillas fijas y registradas; versiones de las dependencias bloqueadas en un archivo de bloqueo.
- Todas las tablas y figuras se generan por script desde los resultados guardados.

## Estructura del repositorio (planificado)

```text
uncertainty-estimation-tfm/
├── pyproject.toml        # dependencias y metadatos
├── uv.lock               # versiones exactas bloqueadas
├── README.md
├── LICENSE
├── CITATION.cff
├── configs/              # configuración por escenario, modelo y método
├── src/uq/
│   ├── data/             # carga y particiones de los datos
│   ├── models/           # MLP y CNN
│   ├── methods/          # los cinco métodos
│   ├── metrics/          # calibración, detección de errores, selectiva y conforme
│   ├── train.py
│   └── evaluate.py
├── scripts/              # ejecución completa, figuras y tablas
├── notebooks/            # análisis exploratorio
├── tests/                # pruebas de métricas y particiones
├── results/              # resultados resumidos
└── docs/                 # notas de método
```

## Instalación y uso (planificado)

El proyecto usa herramientas modernas de Python: gestión del entorno con [`uv`](https://docs.astral.sh/uv/), configuración en `pyproject.toml`, formato y *lint* con `ruff` y pruebas con `pytest`. Se requiere una GPU compatible con CUDA para reproducir el escenario de imágenes en un tiempo razonable; el escenario tabular puede ejecutarse en CPU.

```bash
# 1. Clonar el repositorio
git clone https://github.com/08edu02/uncertainty-estimation-tfm.git
cd uncertainty-estimation-tfm

# 2. Crear el entorno con las versiones bloqueadas
uv sync

# 3. Descargar los datos desde la fuente oficial
uv run python scripts/download_data.py

# 4. Reproducir todos los experimentos, figuras y tablas
uv run python scripts/run_all.py
```

> Los nombres de los scripts son provisionales y pueden cambiar. Las instrucciones definitivas se confirmarán cuando el código esté listo.

## Resultados

*Pendiente.* Esta sección mostrará las tablas comparativas y las figuras principales (diagramas de fiabilidad, curvas riesgo-cobertura y resultados de la predicción conforme) cuando terminen los experimentos.

## Limitaciones y uso responsable

- Los conjuntos de datos son públicos y reducidos. **Los resultados son de investigación: no sustituyen una validación clínica ni deben usarse para decisiones reales de crédito.**
- El conjunto de crédito contiene variables sensibles (como sexo, edad o estado civil), que pueden introducir sesgos. No se extraen conclusiones sociales de este trabajo.
- El estudio se limita a clasificación y no cubre datos fuera de distribución, regresión ni modelos de lenguaje.
- La predicción conforme ofrece una garantía **marginal** (en promedio sobre los datos), no condicional por clase o subgrupo.

## Estado del proyecto

- [ ] Estructura del repositorio y entorno reproducible
- [ ] Análisis exploratorio de los dos conjuntos de datos
- [ ] Métricas implementadas y probadas
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

Cuando el trabajo esté terminado se añadirá un archivo `CITATION.cff`. Mientras tanto, puedes citarlo así:

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

Código bajo licencia **MIT**. Los conjuntos de datos tienen sus propias licencias y no forman parte de este repositorio.

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