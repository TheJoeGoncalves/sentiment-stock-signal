# StockNewsSentiment

A Jupyter Notebook pipeline that fetches financial news headlines, scores them with multiple sentiment models, and trains XGBoost classifiers to predict short-term stock price direction.

The default configuration targets any stock (**NVIDIA (NVDA)** by default) over a time period (April 2024 – April 2026 by default), the ticker, date range, and model toggles are all controlled by a single configuration cell at the top of the notebook.

Currently fixing compatibility with thinking models

---

## How It Works

The pipeline runs end-to-end in a single notebook with the following stages:

**1. Data ingestion** — Headlines are pulled from the [GDELT Project](https://www.gdeltproject.org/) via Google BigQuery, one partition per day. Each day's results are cached to a local Parquet checkpoint so the notebook can resume after interruptions. Only headlines that match the configured keyword list (e.g. `["NVIDIA", "NVDA"]`) are kept.

**2. Sentiment scoring** — Each headline is scored by up to four models, independently toggled via boolean flags:

| Flag | Model | Notes |
|---|---|---|
| `RUN_FINBERT` | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | Finance-tuned BERT; outputs negative / neutral / positive |
| `RUN_VADER` | VADER (`nltk`) | Lexicon-based; fast, no GPU required |
| `RUN_V2TONE` | GDELT V2Tone | Pre-computed tone score from the GDELT record itself |
| `RUN_GEMMA` | GEMMA 3 & 4 (4B) or any Ollama-served model | LLM-based scoring; defaults to Gemma 3 4B & Gemma 4 E4B, but any model available in Ollama can be substituted |

The `LLM_MODELS` list accepts any model ID recognised by your local Ollama installation — swap in `llama3`, `mistral`, `phi3`, `qwen2`, or any other model and it will be pulled automatically if not already present. You can run a single LLM or several in parallel; each gets its own score column and its own downstream XGBoost classifier.

**3. Feature engineering** — Daily headline scores are aggregated into per-model feature sets (rolling average, standard deviation, momentum at 5- and 10-day windows, shock, volume, surprise, acceleration, extreme-score count, flip count). These are joined with price-derived features (RSI, 5- and 20-day momentum, MA-20, MA-50, volume change, 20-day volatility, VIX) fetched from Yahoo Finance.

**4. Binary target** — The target label is whether the stock's *n*-day forward return is positive (`TARGET_HORIZON_DAYS = 1`, next day, by default).

**5. Model training** — A separate XGBoost classifier (`XGBClassifier`) is trained for each sentiment model using a grid search over hyperparameters. An optional dead zone filters out predictions near the decision boundary before evaluation.

**6. Evaluation & diagnostics** — Each trained model is evaluated on held-out validation and test splits. A 4-panel diagnostic grid is produced per model: confusion matrix, ROC curve, probability distribution, and feature importance. Holdout metrics (accuracy, precision, recall, F1) are printed in a summary table.

**7. Trading simulation** — Predictions are converted into a simplified long / short / long–short strategy. Cumulative returns for each strategy and for the raw market are plotted side-by-side.

**8. Session backup** — All DataFrames and trained models are serialised to Parquet and Joblib files in a timestamped backup directory. A plain-text model summary is written alongside. Sessions can be restored from any backup directory.

---

## Requirements

### Python packages

```
pandas
numpy
pyarrow
requests
joblib
matplotlib
seaborn
scikit-learn
xgboost
torch
transformers          # for FinBERT
nltk                  # for VADER
vaderSentiment
yfinance
google-cloud-bigquery
tqdm
```

### External services

| Service | Purpose |
|---|---|
| **Google Cloud / BigQuery** | GDELT headline retrieval |
| **Ollama** (local, `http://127.0.0.1:11434`) | Serving Gemma 3 and Gemma 4 LLMs |
| **Yahoo Finance** | Price and VIX data (via `yfinance`) |

### Environment variables

```
GCP_CREDENTIALS_PATH   # path to your GCP service-account JSON key
GCP_PROJECT_ID         # your Google Cloud project ID
```

---

## Configuration

All user-facing settings live in the first notebook cell:

```python
# ── Target stock ──────────────────────────────────────────────────────────────
TICKER       = "NVDA"
COMPANY_NAME = "NVIDIA"
KEYWORDS     = ["NVIDIA", "NVDA"]   # used for GDELT regex and headline filtering

# ── Date range ────────────────────────────────────────────────────────────────
DATE_FROM = "2024-04-12"
DATE_TO   = "2026-04-15"

# ── Sentiment model toggles ───────────────────────────────────────────────────
RUN_FINBERT = True
RUN_VADER   = True
RUN_V2TONE  = True
RUN_GEMMA   = True

# ── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Add, remove, or swap any Ollama-compatible model.
# Format: (name, ollama_model_id)
# "name" becomes the score column prefix, e.g. "gemma3" -> "gemma3_score"
LLM_MODELS = [
    ("gemma3", "gemma3:4b"),
    ("gemma4", "gemma4:e4b"),
    # ("llama3",  "llama3.1:8b"),
    # ("mistral", "mistral:7b"),
    # ("phi3",    "phi3:mini"),
    # ("qwen",    "qwen2.5:7b"),
]

SIM_MODEL = "gemma4"

# ── BigQuery ──────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), ".env"))
GCP_CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")
GCP_PROJECT_ID       = os.getenv("GCP_PROJECT_ID")


# ── Fetch / checkpoint settings ───────────────────────────────────────────────
FETCH_DELAY_SECONDS = 0.5    # polite pause between BigQuery calls
LLM_MAX_RETRIES   = 3
LLM_RETRY_DELAY   = 2      # seconds between retries
LLM_POLL_INTERVAL = 15     # seconds between model-detection polls
LLM_CHUNK_SIZE    = 500     # number of articles sent per LLM API call
LLM_WORKERS       = 5      # parallel LLM calls (set to CPU cores or less)

# ── Price / feature engineering ───────────────────────────────────────────────
TARGET_HORIZON_DAYS = 1      # n-day forward return for binary target
LOOKBACK_DAYS       = 80     # warm-up days downloaded before DATE_FROM (needs ~55 trading days for ma_50)
ROLLING_WINDOW      = 30     # long-term sentiment baseline window
NEUTRAL_MULTIPLIER  = 0.02   # near-neutral article filter (× rolling std)

# ── Train / val / test split ──────────────────────────────────────────────────
VALIDATION_SPLIT = 0.20      # fraction of pre-test data held for validation

# ── XGBoost grid search ───────────────────────────────────────────────────────
XGB_FEATURES = [
    "avg_score", "momentum", "std", "shock", "volume", "surprise_10", "accel",
]
TARGET_COL          = "target"
DECISION_THRESHOLD  = 0.5    # set to "X" to enable auto-threshold search
THRESHOLD_DEAD_ZONE = (0.0, 0.0)  # predictions in (low, high) are abstained
SELECTION_METRIC    = "precision"  # accuracy | precision | recall | f1 | roc_auc
SCALE_POS_WEIGHT    = False  # True → compensate for class imbalance

XGB_PARAM_GRID = {
    "n_estimators":      [100, 150],
    "max_depth":         [2, 3, 4],
    "min_child_weight":  [3, 5, 7],
    "learning_rate":     [0.01, 0.03, 0.05],
    "subsample":         [0.5, 0.7],
    "colsample_bytree":  [0.5, 0.6],
    "colsample_bylevel": [0.5, 0.7],
    "gamma":             [0.3, 1.0],
    "reg_alpha":         [0.4, 2.0],
    "reg_lambda":        [4, 10],
}

# ── Trading simulation ────────────────────────────────────────────────────────
LONG_THRESHOLD  = 0.55
SHORT_THRESHOLD = 0.45

# ── Paths ─────────────────────────────────────────────────────────────────────
SESSION_LOG_DIR = r"D:\PRODStockNews\session_logs"   # checkpoints, parquets, frozen models
BACKUP_ROOT     = r"D:\PRODStockNews\backup"

```

---


## Features

Two categories of features are built and then joined on date before training.

### Sentiment features (per model)

One set of the following columns is produced for each active sentiment model (e.g. `bert_avg`, `vader_avg`, `gemma3_avg`, …). In the model DataFrames fed to XGBoost the prefix is stripped and replaced with generic names (`avg_score`, `std`, etc.) so the same `XGB_FEATURES` list applies to every model.

| Column suffix | Description |
|---|---|
| `_avg` | Daily mean sentiment score across all headlines |
| `_std` | Daily standard deviation of sentiment scores |
| `_count` | Number of headlines scored that day |
| `_long_term` | Rolling mean of `_avg` over `ROLLING_WINDOW` days (default 30) — the baseline |
| `_shock` | Z-score of today's `_avg` relative to the long-term baseline: `(avg − long_term) / std` |
| `_volume` | Sentiment-weighted article volume: `avg × √count` |
| `_5d_avg` | 5-day rolling mean of `_avg` |
| `_momentum` | Short-term momentum: `avg − 5d_avg` |
| `_10d_avg` | 10-day rolling mean of `_avg` |
| `_momentum_10` | Medium-term momentum: `avg − 10d_avg` |
| `_surprise_5` | Same as `_momentum` (deviation from 5-day average) |
| `_surprise_10` | Same as `_momentum_10` (deviation from 10-day average) |
| `_accel` | Second difference of `_avg` — rate of change of momentum |
| `_extreme` | Binary flag: 1 if `|avg| > 0.5`, indicating an unusually strong sentiment day |
| `_flip` | Binary flag: 1 if sentiment direction reversed vs. the previous day |

A single cross-model feature is also added:

| Column | Description |
|---|---|
| `model_disagreement` | Standard deviation of `_avg` across all active models on a given day — high values indicate models disagree on tone |

### Price & market features (shared across all models)

| Column | Description |
|---|---|
| `rsi` | 14-day Relative Strength Index |
| `momentum_5d` | 5-day price return (Close pct change over 5 days) |
| `momentum_20d` | 20-day price return |
| `ma_20` | 20-day simple moving average of Close |
| `ma_50` | 50-day simple moving average of Close |
| `volume_change` | Day-over-day percentage change in trading volume |
| `volatility_20d` | 20-day rolling standard deviation of daily returns |
| `vix` | CBOE Volatility Index (^VIX) close, merged by date |

### Default XGBoost feature set

The features actually passed to the classifier are controlled by `XGB_FEATURES` in the config cell. The default selection is:

```python
XGB_FEATURES = [
    "avg_score", "momentum", "std", "shock", "volume", "surprise_10", "accel",
]
```

All remaining engineered columns (price features, `_extreme`, `_flip`, `model_disagreement`, etc.) are available in the merged DataFrames and can be added to `XGB_FEATURES` without any other code changes.

---



## Output files

All intermediate and final artefacts are written to `SESSION_LOG_DIR`:

| File | Contents |
|---|---|
| `checkpoints_<TICKER>/<TICKER>_<date>.parquet` | Per-day raw headline cache |
| `df_sentiments_final.parquet` | All scored headlines |
| `xgb_<model>_frozen.pkl` | Trained and frozen XGBoost models |
| `backup_<timestamp>_<top_model>/` | Full session snapshot (DataFrames + models + summary) |

---



## Visualizations

### Model diagnostic grid

Produced for each sentiment model on both the validation and test splits by `plot_model_diagnostics()`. Each model gets a row of four panels:

- **Confusion matrix** — predicted Up/Down vs actual, rendered as a heatmap
- **ROC curve** — with AUC score labelled; falls back to a text notice if only one class is present after dead-zone filtering
- **Probability distribution** — histogram of predicted probabilities, coloured by actual class, showing how well the model separates Up from Down
- **Feature importance** — horizontal bar chart of the top 12 XGBoost feature importances for that model

All five sentiment models are stacked vertically in a single figure (one row per model), making it easy to compare signal quality across FinBERT, VADER, V2Tone, and any LLMs at a glance.

### Trading simulation chart

Produced by `plot_trading_simulation()` after the simulation runs on the test set. A single time-series line chart with four series plotted over the test period:

| Series | Colour | Description |
|---|---|---|
| Long only | Green | Go long when predicted probability ≥ `LONG_THRESHOLD` |
| Short only | Red | Go short when predicted probability ≤ `SHORT_THRESHOLD` |
| Long/Short | Blue | Combine both signals |
| Market | Gray (dashed) | Buy-and-hold benchmark |

The y-axis is cumulative return (starting at 1.0). The model used for simulation is set via `SIM_MODEL` in the config cell.

### Metrics summary table

Printed to the notebook output after evaluation, listing accuracy, precision, recall, and F1 for each model on the validation and holdout test splits. This is a text/DataFrame table rather than a chart, but provides a quick numeric comparison alongside the visual diagnostics.

---


## Resuming a session

The notebook writes a Parquet checkpoint for each calendar day it fetches. On subsequent runs it skips any day already cached, so interrupted runs can be restarted without re-querying BigQuery.

To restore a previous session entirely:

```python
restore_session(r"D:\PRODStockNews\backup\backup_20260415_1801__BERT_100")
```

---

## Notes

- FinBERT requires a CUDA-capable GPU for reasonable throughput on large date ranges; it will fall back to CPU if no GPU is detected.
- Ollama must be running locally before the Gemma scoring cells execute. The notebook polls until it becomes reachable and pulls model weights automatically if they are not already present.
- VADER and V2Tone scoring are CPU-only and complete quickly relative to the transformer-based models.
- The dead zone (`THRESHOLD_DEAD_ZONE`) can be widened to force the classifier to abstain on low-confidence predictions, which typically improves precision at the cost of coverage.
