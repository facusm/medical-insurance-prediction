# MetLife Challenge — Predicción de costos de seguro médico (Charges)

Este repositorio contiene una solución **end-to-end** para predecir el costo de seguro médico (`charges`) usando el dataset provisto (`data/dataset.csv`). La solución incluye:

- Ingesta del CSV en una **base PostgreSQL** (tabla `training_dataset`)
- **Pipeline de entrenamiento** con preprocesamiento + **optimización bayesiana (Optuna/TPE)** + **LightGBM**
- **Pipeline de scoring** sobre una tabla de prueba (10 filas) + reporte de métrica final
- Ejecución **dockerizada** (los pasos 2 → 3 → 4 se ejecutan secuencialmente sin intervención)

---

## 1) Enfoque de modelado (breve justificación)

El problema es de **regresión sobre datos tabulares** con variables numéricas y categóricas, y con potenciales **no linealidades e interacciones** (por ejemplo, `smoker` × `bmi` × `age`). Por este motivo, se eligió un modelo de **Gradient Boosting con árboles** (LightGBM), ya que suele tener muy buen desempeño en datos tabulares y captura relaciones no lineales e interacciones de forma natural. Para seleccionar hiperparámetros de manera eficiente se utiliza **optimización bayesiana** con Optuna (sampler TPE) y validación cruzada.

---

## Estructura del proyecto

```text
metlife_challenge/
│
├── data/
│   └── dataset.csv
│
├── artifacts/                  # Salidas (modelo, métricas)
│   ├── best_model.joblib
│   ├── metrics.txt
│   └── scoring_metrics.txt
│
├── optuna/                     # Opcional: persistir Optuna en volumen
│   └── optuna_study.db
│
├── config/
│   └── db_config.py            # Lee ENV (Docker) o defaults (local)
│
├── src/
│   ├── db/
│   │   └── connection.py       # Engine SQLAlchemy (Postgres)
│   ├── data_access/
│   │   └── load_training_data.py
│   ├── features/
│   │   └── preprocessing.py    # Split + ColumnTransformer (OHE)
│   ├── models/
│   │   ├── train.py            # Optuna + LGBM + CV + resume trials
│   │   └── evaluate.py         # CV mean/std + holdout RMSE
│   └── utils/
│       └── io.py               # Save/load modelo + guardar métricas
│
├── scripts/
│   ├── create_db_and_load_data.py  # Paso 2: crea tablas + carga CSV
│   ├── training.py                 # Paso 3: orquesta train+eval+save
│   └── scoring.py                  # Paso 4: samplea 10 del holdout + predice
│
├── docker/
│   └── entrypoint.sh           # Paso 5: corre 2->3->4 secuencialmente
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```



---

## Mapeo del challenge (requisitos → implementación)

### ✅ Paso 2 — Crear instancia de DB y cargar `dataset.csv` en `training_dataset`
**Script:** `scripts/create_db_and_load_data.py`

- Crea dos tablas:
  - `training_dataset_raw` (staging, tipos similares al CSV)
  - `training_dataset` (tabla final, incluye `id SERIAL PRIMARY KEY` y `smoker BOOLEAN`)
- Carga `data/dataset.csv` en la tabla staging y luego inserta en la tabla final
- Convierte `smoker` de `"yes"/"no"` a boolean con la lógica `smoker = 'yes'`

---

### ✅ Paso 3 — Pipeline de entrenamiento
**Orquestador:** `scripts/training.py`

Realiza lo siguiente:

1. Lee la tabla `training_dataset` desde PostgreSQL
2. Hace un **train/test split** (holdout) con `test_size=0.2`
3. Define el preprocesamiento:
   - Numéricas: passthrough (`age`, `bmi`, `children`)
   - Categóricas: One-Hot Encoding (`sex`, `region`)
   - Binaria: passthrough (`smoker`)
4. Ejecuta **Optuna (TPE)** con **K-Fold CV** sobre el conjunto de train (80%)
5. Entrena el modelo final con los **mejores hiperparámetros** sobre todo el train
6. Evalúa y guarda métricas:
   - `CV_RMSE_best_trial` (valor óptimo encontrado por Optuna)
   - `CV_RMSE_mean` / `CV_RMSE_std` recalculados para los `best_params`
   - `TEST_RMSE` sobre el holdout (20%) **no utilizado** en la optimización
7. Guarda artefactos:
   - `artifacts/best_model.joblib`
   - `artifacts/metrics.txt`
8. Guarda el holdout en la DB como `holdout_test_dataset` para usar luego en `scoring.py`

---

### ✅ Paso 4 — Pipeline de scoring
**Script:** `scripts/scoring.py`

- Lee `holdout_test_dataset` desde PostgreSQL (no usado en Optuna)
- Samplea 10 filas con `random_state=42`
- Carga el modelo entrenado (`artifacts/best_model.joblib`) y predice
- Escribe resultados en la tabla `scoring_results` (incluye `predicted_charges`)
- Reporta RMSE sobre esas 10 filas y lo guarda en `artifacts/scoring_metrics.txt`


---

### ✅ Paso 5 — Docker: empaquetado y ejecución end-to-end
- `Dockerfile` construye una imagen con Python + dependencias
- `docker-compose.yml` levanta:
  - un contenedor `postgres`
  - un contenedor `app` que corre el pipeline
- `docker/entrypoint.sh` orquesta:
  1) Paso 2 (DB + carga CSV)  
  2) Paso 3 (training)  
  3) Paso 4 (scoring)  

---

## Cómo ejecutar: Docker Compose

Desde la **raíz del proyecto**:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from app
```

Esto:

* Inicia Postgres
* Construye la imagen de la app
* Ejecuta secuencialmente los pasos 2 → 3 → 4
* Finaliza con `exit code 0` si todo salió bien

### Re-ejecución “limpia” (borra DB/volúmenes)
Si necesitas reiniciar de cero:

```bash
docker compose down -v
docker compose up --build --abort-on-container-exit --exit-code-from app
```

## Salidas 

Al finalizar la ejecución se generan en la carpeta `artifacts/`:

* `best_model.joblib`: Pipeline (preprocesamiento + modelo) entrenado.
* `metrics.txt`: Métricas del entrenamiento (CV + holdout).
* `scoring_metrics.txt`: Métrica del scoring (sobre 10 filas).
* `optuna_study.db`: Estudio de Optuna persistido (en la carpeta `optuna/`).

## Métrica y evaluación

Se utiliza **RMSE** (Root Mean Squared Error) como métrica principal.

* **Optuna**: Optimiza el RMSE promedio de K-Fold CV solo sobre el train (80%).

* **TEST_RMSE**: Se calcula en un holdout (20%) separado del proceso de tuning.

* **Scoring**: Reporta RMSE en 10 filas muestreadas para validar el flujo end-to-end.

## Dependencias

Ver `requirements.txt`:

* pandas, numpy
* scikit-learn
* SQLAlchemy, psycopg2-binary
* lightgbm
* optuna
* joblib

## Notas

- El repositorio incluye `.gitignore` para evitar subir entornos locales (`.venv/`) y salidas generadas (`artifacts/`, `optuna/`).
- Se incluye `.gitattributes` para forzar finales de línea `LF` en scripts `.sh` y evitar problemas de ejecución en Docker/Linux desde Windows.
