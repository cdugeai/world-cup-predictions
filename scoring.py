import pandas as pd
import numpy as np

# ─── ACTUAL RESULTS LOOKUP ────────────────────────────────────────────────────

def load_actual_results(wc_stats_path: str) -> pd.DataFrame:
    """
    Extract actual results for completed matches from the WC stats file.
    Returns: team1, team2, actual_score, actual_outcome
    """
    df = pd.read_csv(wc_stats_path)
    completed = df[df["status"] == "complete"].copy()

    completed["team1"] = completed["home_team_name"]
    completed["team2"] = completed["away_team_name"]
    completed["actual_score"] = (
        completed["home_team_goal_count"].astype(int).astype(str) + "-" +
        completed["away_team_goal_count"].astype(int).astype(str)
    )

    h, a = completed["home_team_goal_count"], completed["away_team_goal_count"]
    completed["actual_outcome"] = np.where(
        h > a, "team1_win", np.where(h < a, "team2_win", "draw")
    )

    print(f"  Loaded {len(completed)} completed match results")
    return completed[["team1", "team2", "actual_score", "actual_outcome"]]


# ─── SIMPLE JOIN + CORRECTNESS FLAGS ──────────────────────────────────────────

def _score_to_outcome(score_str):
    if pd.isna(score_str):
        return np.nan
    a, b = map(int, str(score_str).split("-"))
    if a > b: return "team1_win"
    if a < b: return "team2_win"
    return "draw"


def add_results(df: pd.DataFrame, actual: pd.DataFrame, score_col: str = "predicted_score") -> pd.DataFrame:
    """
    Left-join actual results onto df (on team1/team2) and add:
      - actual_score, actual_outcome
      - outcome_correct (bool, NaN if not played)
      - score_correct   (bool, NaN if not played)

    `score_col` is the column in df holding the predicted/recommended score
    (e.g. 'predicted_score' for predictions.csv, 'recommended_score' for mpp_bets.csv).
    """
    # Drop any pre-existing result columns (e.g. if this CSV was already scored
    # in a previous run) to avoid merge suffix collisions like 'actual_score_x'
    df = df.drop(columns=["actual_score", "actual_outcome",
                          "outcome_correct", "score_correct"],
                  errors="ignore")

    df = df.merge(actual, on=["team1", "team2"], how="left").reset_index(drop=True)

    predicted_outcome = df[score_col].apply(_score_to_outcome)
    has_result = df["actual_score"].notna()

    df["outcome_correct"] = pd.Series(np.nan, index=df.index, dtype=object)
    df["score_correct"]   = pd.Series(np.nan, index=df.index, dtype=object)

    df.loc[has_result, "outcome_correct"] = (
        predicted_outcome[has_result] == df.loc[has_result, "actual_outcome"]
    )
    df.loc[has_result, "score_correct"] = (
        df.loc[has_result, score_col] == df.loc[has_result, "actual_score"]
    )

    n = has_result.sum()
    if n > 0:
        print(f"  {n} matches scored — outcome accuracy "
              f"{df.loc[has_result,'outcome_correct'].mean():.1%}, "
              f"exact score {df.loc[has_result,'score_correct'].mean():.1%}")

    return df