# MLOps Architecture: Supply-Chain Fulfillment Risk System

## 1. Business Goal

Build a production-grade, human-in-the-loop ML system that predicts fulfillment risk for supply-chain operations.

The system predicts the expected fulfillment rate for a sales order line, explains the risk in operational terms, and routes uncertain or high-impact cases to human reviewers before action is taken.

## 2. ML Formulation

- **Problem type**: Regression
- **Prediction unit**: One sales order line
- **Target**: `fulfillment_rate = Quantity_Invoiced / Quantity_Ordered`
- **Output range**: 0.0 to 1.0
- **Primary business use**: Help operations and procurement teams identify orders/products likely to be under-fulfilled.

The predicted fulfillment rate is converted into operational risk bands:

| Risk Band | Fulfillment Rate |
|---|---:|
| Critical | `< 0.25` |
| High | `0.25 - 0.50` |
| Medium | `0.50 - 0.75` |
| Low | `0.75 - 0.99` |
| OK | `>= 0.99` |

## 3. MLOps Maturity Target

The v1 target is **Level 1 MLOps with selected Level 2 governance controls**.

Included in v1:

- Reproducible training pipeline
- Strict data validation
- Versioned artifacts
- Experiment tracking
- Model registry
- Batch inference
- Human review gates
- Promotion approval before production use
- Monthly monitoring and retraining policy

Deferred until later:

- Fully automated retraining and deployment
- Real-time serving API
- Kubernetes
- Feature store
- Cloud-native CI/CD
- Automated alerting

## 4. Source Data

Current source files:

| File | Rows | Purpose |
|---|---:|---|
| `data/Product_Inventory.csv` | 1,000 | Product, inventory, price, cost, vendor, class, responsibility |
| `data/Purchase_Orders.csv` | 1,000 | Vendor/product purchase orders, quantities, dates, receipt performance |
| `data/Sales_Invoiced.csv` | 1,000 | Customer sales orders and invoiced fulfillment |

Current update frequency:

- Monthly batch updates

## 5. Data Plan

### Ingestion

v1 uses monthly batch ingestion from CSV files.

The pipeline reads:

- Product inventory
- Purchase orders
- Sales invoiced records

### Data Versioning

Each pipeline run should preserve:

- Raw input snapshot
- Validated dataset artifact
- Feature dataset artifact
- Training dataset split
- Model artifact
- Evaluation report
- Batch prediction output

### Validation Gates

The data validation step should fail fast if critical assumptions are broken.

Required checks:

- Required columns exist
- Numeric quantities and prices are non-negative
- Dates parse correctly
- `Delivery_Date >= Order_Date`
- `Quantity_Invoiced <= Quantity_Ordered`
- `Qty_Received <= Qty_Ordered`, unless over-receipt is explicitly allowed
- Sales products join to inventory products
- Sales products join to purchase-order products
- Purchase-order vendors join to inventory vendors
- Leakage columns are excluded from model features

Known data caveat:

`Sales_Invoiced.csv` currently has no sales or invoice date. That limits production safety because the pipeline cannot prove which inventory and purchase-order facts were known at prediction time. Before true production use, the sales source should include an order date or invoice date.

## 6. Feature Engineering Plan

### Excluded Leakage Columns

The model must not use columns that reveal the answer.

Exclude from input features:

- `Quantity_Invoiced`
- `fulfillment_rate`
- `risk_band`
- `Qty_Received`
- `Purchase_Orders.Total`
- Future delivery outcomes
- Post-sale inventory states

### Candidate Input Features

Sales-order features:

- `Product`
- `Pharmacy_Name`
- `Quantity_Ordered`
- `Unit_Price`
- `order_value = Quantity_Ordered * Unit_Price`

Product and inventory features:

- `Product_Category`
- `Sales_Price`
- `Cost`
- `margin = Sales_Price - Cost`
- `margin_pct`
- `Qty_On_Hand`
- `Lagos`
- `Ibadan`
- `total_location_stock`
- `zero_stock_flag`
- `Class`
- `Responsible`
- `Vendors`

Historical purchase-order features by product/vendor:

- Average ordered quantity
- Average receipt rate
- Average lead time
- Short-receipt frequency
- Vendor-level receipt performance

### Preprocessing

Use a single bundled preprocessing and model pipeline.

Numeric features:

- Median imputation
- No scaling required for tree-based models

Categorical features:

- Missing value imputation with `UNKNOWN`
- One-hot encoding
- Handle unseen production categories safely

The same preprocessing code must be used during training and inference to prevent training-serving skew.

## 7. Training Plan

### Baselines

Baseline models:

- Mean fulfillment-rate predictor
- Simple decision tree regressor

These baselines are required because a candidate model is only useful if it beats a simple, honest reference point.

### Candidate Models

Initial candidates:

- Random forest regressor
- Histogram gradient boosting regressor

These are suitable for tabular data, handle nonlinear relationships, and do not require heavy feature scaling.

### Experiment Tracking

Use MLflow to track:

- Run ID
- Data version
- Feature version
- Model type
- Hyperparameters
- Metrics
- Artifacts
- Model version

### Split Strategy

v1:

- Holdout train/test split

Future production improvement:

- Time-based split once sales order dates are available

## 8. Evaluation Plan

Primary metric:

- **MAE**: Mean Absolute Error

MAE is the average absolute difference between predicted and actual fulfillment rate. It is easy for business users to understand. An MAE of `0.12` means the model is wrong by about 12 fulfillment-rate percentage points on average.

Guardrail metrics:

- RMSE
- R2
- Risk-band accuracy
- Critical/High recall
- Error by product category
- Error by vendor
- Error by pharmacy

Human-in-the-loop metrics:

- Review precision
- Review queue volume
- Human override rate
- Human approval rate
- Final outcome after review

Initial promotion gate:

- Candidate model beats mean baseline MAE by at least 10%
- Critical/High recall is acceptable to the business
- Review queue volume is operationally manageable
- No critical data validation failures
- Human approval before production promotion

## 9. Deployment Plan

v1 deployment mode:

- Monthly batch inference

Monthly flow:

1. New CSV files arrive.
2. Validation pipeline checks schema and business rules.
3. Feature pipeline builds model-ready records.
4. Batch inference scores records.
5. Output includes predicted fulfillment rate, risk band, explanation, and review flag.
6. Human reviewers inspect flagged cases.
7. Human decisions are stored for audit and future training.

Prediction output schema:

- `SO_Number`
- `Product`
- `Pharmacy_Name`
- `Quantity_Ordered`
- `Unit_Price`
- `predicted_fulfillment_rate`
- `predicted_risk_band`
- `estimated_shortfall_value`
- `review_required`
- `review_reason`
- `model_version`
- `prediction_timestamp`

## 10. Human-In-The-Loop Design

The model should recommend, not automatically execute business actions.

Review is required when:

- Predicted fulfillment rate is below 50%
- Estimated shortfall value is high
- Model confidence is low
- Product has zero or very low inventory
- Vendor receipt history is poor
- Data validation produced a warning

Human reviewers should capture:

- Approved action
- Rejected action
- Changed quantity
- Manual reason
- Vendor follow-up needed
- Data issue flag
- Final outcome

These decisions become feedback data for future model improvement.

## 11. Monitoring and Drift Plan

Monitoring cadence:

- Monthly

Track data quality:

- Missing columns
- Bad date parse rate
- Negative quantity count
- Product join failure rate
- Vendor join failure rate

Track prediction behavior:

- Average predicted fulfillment rate
- Risk-band distribution
- Number of human-review cases
- Percentage of records routed to review

Track model performance:

- MAE after actual invoiced quantities are known
- Critical/High recall
- Error by product category
- Error by vendor
- Error by pharmacy

Track drift:

- Product mix changes
- Product category mix changes
- Quantity ordered distribution changes
- Unit price distribution changes
- Vendor receipt-rate distribution changes

Retraining triggers:

- Monthly scheduled retraining
- MAE worsens by more than 15%
- Product mix changes materially
- Review precision drops
- Enough new human decisions accumulate

## 12. Governance and Auditability

Every prediction should be auditable.

Store:

- Input data version
- Feature dataset version
- Model version
- Prediction timestamp
- Predicted fulfillment rate
- Predicted risk band
- Review decision
- Reviewer role or ID
- Human override reason
- Final outcome

Promotion workflow:

| Stage | Meaning |
|---|---|
| Development | Model trained and evaluated locally |
| Staging | Model passed validation and metric gates |
| Production | Model approved for batch decision support |

Production promotion requires human approval.

## 13. Recommended ZenML Stack

| Component | v1 Choice | Reason |
|---|---|---|
| Orchestrator | ZenML local | Sufficient for monthly batch v1 |
| Artifact store | Local filesystem | Simple and inspectable |
| Experiment tracker | MLflow | Tracks runs, metrics, parameters, and artifacts |
| Model registry | MLflow | Provides model versioning and promotion path |
| Data validation | Custom checks first | Business rules are specific and simple |
| Deployer | Batch inference script/pipeline | Real-time API is not needed for v1 |
| Monitoring | Monthly reports/artifacts | Matches monthly update cadence |
| Container registry | None | Add when moving to cloud or CI/CD |
| Alerter | None | Add email or Slack later if needed |
| Step operator | None | Local execution is enough for v1 |

## 14. Pipeline Decomposition

### Training Pipeline

1. Load raw CSVs
2. Validate schemas and business rules
3. Build features
4. Create target
5. Split data
6. Train baseline
7. Train candidate model
8. Evaluate
9. Register model if gates pass

### Batch Inference Pipeline

1. Load latest approved model
2. Load new monthly data
3. Validate input data
4. Build inference features
5. Predict fulfillment rate
6. Convert prediction to risk band
7. Apply human-review rules
8. Save prediction output

### Monitoring Pipeline

1. Load latest predictions
2. Compare current data to reference data
3. Compute drift metrics
4. Compute outcome metrics when actuals are available
5. Produce monthly monitoring report
6. Flag retraining or investigation if thresholds are exceeded

### Retraining Pipeline

1. Trigger monthly or from monitoring signal
2. Re-run training pipeline
3. Compare candidate model to production model
4. Promote only if gates pass
5. Require human approval for production promotion

## 15. Proposed Project Structure

```text
.
├── architecture.md
├── data/
│   ├── Product_Inventory.csv
│   ├── Purchase_Orders.csv
│   └── Sales_Invoiced.csv
├── configs/
├── core/
│   ├── validation.py
│   ├── features.py
│   ├── evaluation.py
│   └── review_rules.py
├── steps/
├── pipelines/
├── reports/
├── artifacts/
├── tests/
└── README.md
```

`core/` should contain pure Python logic with no ZenML-specific imports. Framework-specific steps should be thin wrappers around `core/`. This keeps the important logic testable and easier to migrate later.

## 16. MVP Scope

Build first:

1. Project structure
2. Data validation
3. Feature-building pipeline
4. Baseline model
5. Candidate regression model
6. Evaluation report
7. Batch prediction output
8. Human-review flag logic
9. Model and artifact tracking

## 17. Deferred Components

Defer until the MVP proves value:

- Real-time API
- Feature store
- Kubernetes
- Automated cloud deployment
- Automated retraining without human approval
- SHAP explanations
- Slack/email alerting
- Full CI/CD

## 18. Open Decisions

These decisions should be made during implementation:

- Exact threshold for high-value shortfall review
- Acceptable monthly human-review queue size
- Minimum acceptable Critical/High recall
- Whether over-receipt should ever be allowed in purchase-order data
- Where human decisions will be stored
- Whether the next data extract can include sales order or invoice dates

