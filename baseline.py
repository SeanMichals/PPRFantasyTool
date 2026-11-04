import nflreadpy as nfl
import pandas as pd

#Load player statistics from 2000 to 2025
stats = nfl.load_player_stats(seasons=list(range(2000, 2026)))
df = stats.to_pandas()
print(df.columns.tolist())

df = df[(df["season_type"] == "REG") & (df["position"] == "RB")]
df = df.sort_values(["player_id", "season", "week"])

#Calculate the baseline fantasy points for each player based on the previous 3 weeks
#shift the fantasy points by 1 week to avoid including the current week in the baseline calculation
df["baseline"] = (
    df.groupby("player_id")["fantasy_points_ppr"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
)

df = df.dropna(subset=["baseline"])

test = df[df["season"] == 2025]
print(f"Test rows: {len(test)}")
mae = (test["fantasy_points_ppr"] - test["baseline"]).abs().mean()
print(f"Baseline Mae: {mae:.2f} fantasy points")