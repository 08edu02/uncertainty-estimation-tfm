# EDA mínimo

Generado por `scripts/run_eda.py`; no editar a mano.

### Escenario tabular (crédito)

- Filas: 30,000 · variables: 23
- Duplicados exactos (sin la etiqueta): 56
- Valores faltantes: 0
- Tasa de impago: 0.2212 (6,636 de 30,000)

| Grupo | n | Tasa de impago |
| --- | ---: | ---: |
| sex = 1 | 11,888 | 0.2417 |
| sex = 2 | 18,112 | 0.2078 |
| education = 0 | 14 | 0.0000 |
| education = 1 | 10,585 | 0.1923 |
| education = 2 | 14,030 | 0.2373 |
| education = 3 | 4,917 | 0.2516 |
| education = 4 | 123 | 0.0569 |
| education = 5 | 280 | 0.0643 |
| education = 6 | 51 | 0.1569 |
| marriage = 0 | 54 | 0.0926 |
| marriage = 1 | 13,659 | 0.2347 |
| marriage = 2 | 15,964 | 0.2093 |
| marriage = 3 | 323 | 0.2601 |

### Escenario de imágenes (PneumoniaMNIST)

| Conjunto | Imágenes | Clase 0 | Clase 1 | Proporción clase 1 |
| --- | ---: | ---: | ---: | ---: |
| train | 4,708 | 1,214 | 3,494 | 0.7421 |
| val | 524 | 135 | 389 | 0.7424 |
| test | 624 | 234 | 390 | 0.6250 |

- Forma de las imágenes: [4708, 28, 28] (uint8), píxeles en [0, 255]
- Media de píxel: 145.84 ± 42.93

## Figuras

- `credit_class_balance.png`
- `credit_distributions.png`
- `pneumonia_split_balance.png`
- `pneumonia_samples.png`
