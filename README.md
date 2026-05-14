# Supply-Chain Fulfillment Risk MLOps System

This project builds a production-oriented, human-in-the-loop ML system for predicting sales-order fulfillment rate from supply-chain tabular data.

The first production target is monthly batch decision support:

1. Validate monthly source data.
2. Build leakage-safe features.
3. Train and evaluate regression models.
4. Predict fulfillment rate.
5. Convert predictions into risk bands.
6. Route high-risk or uncertain cases to human reviewers.
7. Track decisions, outcomes, model versions, and monitoring signals.

See `architecture.md` for the full system design.

## Current Data

Source files live in `data/`:

- `Product_Inventory.csv`
- `Purchase_Orders.csv`
- `Sales_Invoiced.csv`

## Target

```text
fulfillment_rate = Quantity_Invoiced / Quantity_Ordered
```

The model predicts fulfillment rate as a regression problem, then maps the prediction to operational risk bands.

## Project Layout

```text
configs/      Runtime settings and business thresholds
core/         Pure Python validation, features, evaluation, and review logic
steps/        Future ZenML step wrappers
pipelines/    Future ZenML pipeline definitions
tests/        Unit and integration tests
reports/      Evaluation and monitoring reports
artifacts/    Local generated artifacts
```

## Status

Scaffold created. Next implementation step: data validation.

## Local Training

After Python and project dependencies are installed, run:

```powershell
python scripts/train.py
```

The script validates the source CSVs, builds the training dataset, trains a mean baseline and candidate regression model, then writes:

```text
reports/training_metrics.json
artifacts/model.joblib
```

## Batch Inference

After a model has been trained, run:

```powershell
python scripts/predict_batch.py
```

The script validates the latest source CSVs, builds leakage-safe inference features, scores each sales order line, applies human-review rules, and writes:

```text
reports/batch_predictions.csv
```

## Dependency-Free Fallback

If scientific Python packages cannot be installed yet, use the standard-library fallback:

```powershell
.venv\Scripts\python.exe scripts/train_stdlib.py
.venv\Scripts\python.exe scripts/predict_batch_stdlib.py
```

This trains a transparent product-level mean fulfillment model and writes:

```text
reports/training_metrics_stdlib.json
artifacts/model_stdlib.json
reports/batch_predictions_stdlib.csv
```

## Diagnostics

After training and batch inference, run:

```powershell
.venv\Scripts\python.exe scripts\diagnose_model.py
```

The script analyzes model metrics, target distribution, feature signal, high-error slices, review queue volume, and missing data fields. It writes:

```text
reports/model_diagnostics.json
```

## Staging Deployment

To create a governed staging deployment bundle, run:

```powershell
.venv\Scripts\python.exe scripts\deploy_staging.py
```

This copies the trained model, metrics, and config into `artifacts/staging/` and writes:

```text
artifacts/staging/deployment_manifest.json
```

Staging deployment is allowed even when the production promotion gate fails. Production promotion remains blocked until the candidate model clears the agreed metric gate and receives human approval.

## Human Review Queue

After batch inference, create the human review queue:

```powershell
.venv\Scripts\python.exe scripts\create_review_queue.py
```

This filters `reports/batch_predictions.csv` to review-required rows and writes:

```text
reports/human_review_queue.csv
```

Reviewers should fill in the human decision fields. Those decisions become feedback data for future monitoring and retraining.

## Staging API

The API is for staging/shadow use only. It returns recommendations and review-routing decisions; it must not trigger automatic procurement or inventory actions.

Install API dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[api]" --no-build-isolation
```

Run the API:

```powershell
.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```text
POST http://127.0.0.1:8000/predict
```
