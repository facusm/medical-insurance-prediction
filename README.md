# Predicción de Costos de Seguro Médico: End-to-End Pipeline

Este repositorio contiene una solución automatizada **end-to-end** para predecir el costo de seguros médicos (`charges`) utilizando datos históricos. La arquitectura está diseñada para simular un entorno de producción, abarcando desde la ingesta de datos en bases relacionales hasta el entrenamiento y despliegue del modelo.

## Arquitectura de la Solución

El flujo de trabajo se divide en los siguientes componentes principales:
- Ingesta de datos crudos en una **base de datos PostgreSQL**.
- **Pipeline de entrenamiento** automatizado que incluye preprocesamiento, **optimización bayesiana (Optuna/TPE)** y entrenamiento de un modelo **LightGBM**.
- **Pipeline de scoring (inferencia)** para realizar predicciones sobre nuevos lotes de datos y generar reportes de métricas.
- Entorno de ejecución completamente **dockerizado** para garantizar reproducibilidad.

---

## 1. Enfoque de Modelado

El problema se abordó como una tarea de **regresión sobre datos tabulares** con variables numéricas y categóricas. Dado el potencial de no linealidades e interacciones complejas en este dominio (por ejemplo, `smoker` × `bmi` × `age`), se implementó un modelo de **Gradient Boosting** (LightGBM). Este algoritmo destaca en datos tabulares y captura interacciones de forma nativa. 

Para asegurar la robustez del modelo y una selección eficiente de hiperparámetros, se integró **optimización bayesiana** con Optuna (sampler TPE) evaluada mediante validación cruzada (K-Fold CV).

---

## 2. Estructura del Proyecto

```text
medical_insurance_prediction/
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
│   ├── create_db_and_load_data.py  # Fase 1: crea tablas + carga CSV
│   ├── training.py                 # Fase 2: orquesta train+eval+save
│   └── scoring.py                  # Fase 3: samplea 10 del holdout + predice
│
├── docker/
│   └── entrypoint.sh           # Orquesta ejecución secuencial (Fases 1 -> 2 -> 3)
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```


## 3. Fases del Pipeline

### Fase 1: Ingesta de Datos (`create_db_and_load_data.py`)
- Crea el esquema en PostgreSQL: tabla *staging* y tabla final de producción.
- Carga los datos crudos, normaliza tipos de datos y transforma variables lógicas (ej. `smoker` a formato booleano nativo).

### Fase 2: Entrenamiento y Optimización (`training.py`)
1. Extrae los datos de entrenamiento directamente desde PostgreSQL.
2. Aplica un split de *Holdout* aislando un 20% de los datos para validación final.
3. Define el preprocesamiento de características (One-Hot Encoding para categóricas, *passthrough* para numéricas).
4. Ejecuta **Optuna (TPE)** con **K-Fold CV** sobre el conjunto de entrenamiento.
5. Re-entrena el modelo LightGBM final utilizando los mejores hiperparámetros encontrados.
6. Persiste los artefactos del modelo (`best_model.joblib`) y registra las métricas de rendimiento (`RMSE`).
7. Guarda el set de *Holdout* en la base de datos como `holdout_test_dataset` para simulaciones de inferencia futuras.

### Fase 3: Inferencia y Scoring (`scoring.py`)
- Simula un entorno de producción extrayendo muestras aleatorias del set de prueba desde PostgreSQL (datos no vistos por Optuna).
- Carga el modelo pre-entrenado y genera predicciones (`predicted_charges`).
- Registra los resultados en una nueva tabla de la base de datos (`scoring_results`) y exporta las métricas de rendimiento finales.



## 4. Despliegue y Ejecución (Docker Compose)

El proyecto está diseñado para ejecutarse secuencialmente sin intervención manual. Desde la **raíz del proyecto**, ejecuta:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from app
```

Esto:

* Inicia la instancia de PostgreSQL.
* Construye la imagen de la aplicación (Python + dependencias).
* Ejecuta secuencialmente las Fases 1 → 2 → 3.
* Finaliza con `exit code 0` si todo salió bien

### Re-ejecución “limpia” (borra DB/volúmenes)
Si necesitas reiniciar de cero:

```bash
docker compose down -v
docker compose up --build --abort-on-container-exit --exit-code-from app
```

## 5. Salidas Generadas 

Al finalizar la ejecución se generan en la carpeta `artifacts/`:

* `best_model.joblib`: Pipeline (preprocesamiento + modelo) entrenado.
* `metrics.txt`: Métricas del entrenamiento (CV + holdout).
* `scoring_metrics.txt`: Métrica del scoring (sobre 10 filas).
* `optuna_study.db`: Estudio de Optuna persistido (en la carpeta `optuna/`).

## 6. Métrica y evaluación

Se utiliza **RMSE** (Root Mean Squared Error) como métrica principal.

* **Optuna**: Optimiza el RMSE promedio de K-Fold CV solo sobre el train (80%).

* **TEST_RMSE**: Se calcula en un holdout (20%) separado del proceso de tuning.

* **Scoring**: Reporta RMSE en 10 filas muestreadas para validar el flujo end-to-end.

## 7. Dependencias y Tecnologías

Ver `requirements.txt`:

* pandas, numpy
* scikit-learn
* SQLAlchemy, psycopg2-binary
* lightgbm
* optuna
* joblib

## 8. Notas Técnicas

- El repositorio incluye `.gitignore` para evitar subir entornos locales (`.venv/`) y salidas generadas (`artifacts/`, `optuna/`).
- Se incluye `.gitattributes` para forzar finales de línea `LF` en scripts `.sh` y evitar problemas de ejecución en Docker/Linux desde Windows.
