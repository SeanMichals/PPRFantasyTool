# PPRFantasyTool

A machine learning project that predicts a running back's PPR fantasy points for the next game, using historical NFL data. The goal is to beat a simple "average of the last 3 games" baseline.

## Results

Evaluated on the 2023, 2024, and 2025 regular seasons. For each test season, the model is trained only on earlier seasons so no future data leaks in.

| Model | Mean Absolute Error (PPR points) |
|---|---|
| Baseline: average of last 3 games | 4.712 |
| Gradient boosting, rolling stats only | 4.602 |
| + opponent defense | 4.584 |
| + efficiency stats | 4.584 |
| + Vegas lines | 4.561 |
| **Final model (all of the above)** | **4.548** |

The final model is about 0.16 points (roughly 3.5%) better than the baseline. Weekly football scoring is very noisy (touchdowns, injuries, and game script are hard to predict), so a small, consistent improvement is a realistic result.

## Approach

1. **Data:** Weekly player stats and schedules from [nflverse](https://github.com/nflverse) via the `nflreadpy` package (2012-2025 regular seasons, RBs only).
2. **Baseline:** Predict each player's score as the mean of their previous 3 games.
3. **Features** (all built from information available before the game):
   - Rolling 3-game and 6-game averages of PPR points, carries, targets, receptions, rushing/receiving yards, and target share
   - Opponent defense: average PPR points the opponent allowed to RBs over its previous 6 games
   - Efficiency: rolling rushing EPA, 20+ yard runs, and rushing first downs
   - Vegas lines: team spread and implied team total
4. **Model:** scikit-learn `HistGradientBoostingRegressor`.
5. **Validation:** Time-based splits (train on earlier seasons, test on a later one), averaged across three test seasons.

Features that did not help and were dropped: home/away and week of season.

## Project Structure

```
FFP/
├── baseline.py        # Last-3-games baseline
├── model.py           # Feature experiments and model evaluation
├── dependencies/
│   └── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd FFP
python3 -m venv venv
source venv/bin/activate
pip install -r dependencies/requirements.txt
```

## Usage

```bash
python baseline.py   # prints the baseline MAE
python model.py      # compares the baseline and each feature set
```

The first run downloads data from nflverse, so it needs an internet connection.

## Roadmap

- [ ] Add snap share (`load_snap_counts`)
- [ ] Add injury status (`load_injuries`)
- [ ] Tune model hyperparameters
- [ ] Extend to WR, QB, and TE
- [ ] Compare predictions against ESPN weekly projections
- [ ] Streamlit app: pick two players and get a start/sit recommendation

## Limitations

- Touchdowns and game script make weekly scoring inherently noisy.
- Only past-game data is used, so late injury news and lineup changes are not captured.
- Evaluated on three seasons, so small differences between feature sets may be within noise.

## Data

Player stats and schedules come from [nflverse](https://github.com/nflverse), accessed through `nflreadpy`.
