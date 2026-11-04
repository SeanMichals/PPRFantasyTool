import nflreadpy as nfl
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

target = "fantasy_points_ppr"
seasons = list(range(2012, 2026))

df = nfl.load_player_stats(seasons=seasons).to_pandas()
df = df[(df["season_type"] == "REG") & (df["position"] == "RB")]
df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

def add_rolling(df, cols, windows=(3, 6)):
    names = []
    g = df.groupby("player_id")
    for col in cols:
        for w in windows:
            name = f"{col}_roll{w}"
            df[name] = g[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=3).mean()
            )
            names.append(name)
    return names

# ---- Feature groups (each is a list of column names) ----
base = add_rolling(df, [target, "carries", "targets", "receptions",
                        "rushing_yards", "receiving_yards", "target_share"])

# 1. Opponent defense: avg PPR points that team allowed to RBs in its previous 6 games
allowed = (df.groupby(["opponent_team", "season", "week"])[target]
             .sum().reset_index()
             .sort_values(["opponent_team", "season", "week"]))
allowed["opp_rb_allowed"] = allowed.groupby("opponent_team")[target].transform(
    lambda s: s.shift(1).rolling(6, min_periods=3).mean()
)
df = df.merge(allowed[["opponent_team", "season", "week", "opp_rb_allowed"]],
              on=["opponent_team", "season", "week"], how="left")
opp = ["opp_rb_allowed"]

# 2. Home/away from the schedule
sched = nfl.load_schedules(seasons=seasons).to_pandas()[["game_id", "home_team"]]
df = df.merge(sched, on="game_id", how="left")
df["is_home"] = (df["team"] == df["home_team"]).astype(float)
df.loc[df["home_team"].isna(), "is_home"] = float("nan")
home = ["is_home"]

# 3. Efficiency: rolling EPA and 20+ yard runs
eff = add_rolling(df, ["rushing_epa", "rushing_20", "rushing_first_downs"])

# 4. Week of season (captures early/late season patterns)
wk = ["week"]

# Keep the exact same rows for every experiment so the scores are comparable
df = df.dropna(subset=base).reset_index(drop=True)

def evaluate(features, test_seasons=(2023, 2024, 2025)):
    scores = []
    for ts in test_seasons:
        train, test = df[df["season"] < ts], df[df["season"] == ts]
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, random_state=0)
        model.fit(train[features], train[target])
        scores.append(mean_absolute_error(test[target], model.predict(test[features])))
    return sum(scores) / len(scores)

# Baseline: last-3-game average, scored on the same seasons
b = [mean_absolute_error(df[df.season == s][target],
                         df[df.season == s][f"{target}_roll3"])
     for s in (2023, 2024, 2025)]
print(f"Baseline (last 3 avg): {sum(b)/len(b):.3f}")

experiments = {
    "base rolling stats": base,
    "+ opponent defense": base + opp,
    "+ home/away":        base + home,
    "+ efficiency":       base + eff,
    "+ week of season":   base + wk,
    "all combined":       base + opp + home + eff + wk,
}
for name, feats in experiments.items():
    print(f"{name:22s} MAE: {evaluate(feats):.3f}")
