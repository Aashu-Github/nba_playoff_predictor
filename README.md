# nba_playoff_predictor

# 🏀 NBA Playoff Game Predictor

A machine learning pipeline that predicts the winner of NBA playoff games using 10 seasons of team performance data. Built with **Logistic Regression** trained on differential stats (Offensive Rating, Defensive Rating, Pace, eFG%, 3P%, Rest Days, and more).

---

## 📊 Model Overview

| Detail | Value |
|---|---|
| Model | Logistic Regression (scikit-learn) |
| Features | ~22 differential stats (Home − Away) |
| Training Data | 2014-15 through 2021-22 playoffs |
| Test Data | 2022-23 and 2023-24 playoffs (temporal split) |
| Data Source | [nba_api](https://github.com/swar/nba_api) |

**Key features used:**
- `DIFF_OFF_RTG_PROXY` — Offensive rating differential
- `DIFF_DEF_RTG_PROXY` — Defensive rating differential
- `DIFF_PACE_PROXY` — Pace differential
- `DIFF_EFG_PCT` — Effective field goal % differential
- `DIFF_FG3_PCT` — Three-point % differential
- `DIFF_TOV_RATE` — Turnover rate differential
- `DIFF_REST_DAYS` — Rest day advantage
- `DIFF_ROLL_PLUS_MINUS` — Rolling 5-game plus/minus differential

---

## 🗂️ Project Structure

```
nba-playoff-predictor/
├── data/                    # CSVs (gitignored)
├── models/                  # Saved model + eval charts
├── visualizations/          # Seaborn charts
├── src/
│   ├── fetch_data.py        # Pull 10 seasons of playoff logs via nba_api
│   ├── preprocess.py        # Feature engineering & game-pair construction
│   ├── train.py             # Train, cross-validate, evaluate
│   ├── predict.py           # CLI predictor for any two teams
│   └── visualize.py         # Correlation heatmap, ROC curve, distributions
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

**Python 3.10+ recommended**

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nba-playoff-predictor.git
cd nba-playoff-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create necessary directories
mkdir -p data models visualizations
```

---

## 🚀 Usage

Run the pipeline in order:

```bash
# Step 1 — Fetch 10 seasons of playoff data (~5-10 min, respects API rate limits)
python src/fetch_data.py

# Step 2 — Engineer features and build model-ready dataset
python src/preprocess.py

# Step 3 — Train the model and evaluate on held-out test seasons
python src/train.py

# Step 4 — Generate charts and visualizations
python src/visualize.py

# Step 5 — Predict a specific matchup
python src/predict.py --home "Boston Celtics" --away "Miami Heat"
python src/predict.py --home "Denver Nuggets" --away "Minnesota Timberwolves" --home-rest 3 --away-rest 2
```

---

## 📈 Sample Output

```
====================================================
  🏀  NBA PLAYOFF GAME PREDICTOR
====================================================
  Boston Celtics                    (Home)
  vs.
  Miami Heat                        (Away)
────────────────────────────────────────────────────
  Home win probability : 67.3%
  Away win probability : 32.7%
────────────────────────────────────────────────────
  Predicted winner     : Boston Celtics
====================================================
```

---

## 📉 Visualizations

The `visualize.py` script generates four charts saved to `visualizations/`:

- **Correlation Heatmap** — pairwise correlations between all differential features
- **Feature Distributions** — how each key stat differs between wins and losses
- **ROC Curve** — model discrimination performance (AUC)
- **Rest Day Win Rate** — historical home win rate bucketed by rest day advantage

---

## 🔍 Methodology

### Why Temporal Split?
Sports data has strong temporal structure. Randomly shuffling games would leak future information into training — inflating accuracy. Instead, we train on earlier seasons and test strictly on the most recent two seasons.

### Why Differentials?
Absolute team stats are noisy across eras (pace changed dramatically from 2014 to 2024). Home−Away differentials capture relative team quality and are era-agnostic.

### Proxy Stats
`nba_api`'s `LeagueGameLog` doesn't include advanced stats like Offensive Rating directly. We compute proxies:
- **OffRtg proxy** = `PTS / Possessions × 100`  
- **Possessions** = `FGA − OREB + TOV + 0.44 × FTA`
- **Pace proxy** = estimated possessions per game
- **eFG%** = `(FGM + 0.5 × FG3M) / FGA`

These are the same formulas used by Basketball Reference.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `nba_api` | Pull game logs and team stats from the NBA Stats API |
| `pandas` | Data cleaning, feature engineering, merging |
| `scikit-learn` | Logistic Regression, StandardScaler, Pipeline, cross-validation |
| `seaborn` | Correlation heatmaps, distribution plots, ROC curves |
| `matplotlib` | Underlying chart rendering |
| `joblib` | Model serialization |

---

## 📌 Notes

- **Rate limiting**: `nba_api` requires ~0.7s between requests. Full data fetch takes 5–10 minutes.
- **Data directory**: Raw CSVs are gitignored (too large). Re-run `fetch_data.py` to regenerate.
- **Model file**: `playoff_predictor.pkl` is also gitignored. Re-run `train.py` to regenerate.

---

## 📄 License

MIT License — free to use and modify.
