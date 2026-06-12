import pandas as pd
import numpy as np
from scipy.stats import poisson

# ─── LOAD & CLEAN STATS FILES ────────────────────────────────────────────────

def load_match_stats(*filepaths: str) -> pd.DataFrame:
    """
    Load one or more footystats-style CSVs (friendlies, WC matches, etc.)
    and concatenate into a single dataframe of completed matches.

    Keeps only matches with status == 'complete' (so xg/stats are real, not placeholders).
    """
    frames = []
    for fp in filepaths:
        df = pd.read_csv(fp)
        df["source_file"] = fp.split("/")[-1]
        frames.append(df)

    all_matches = pd.concat(frames, ignore_index=True)

    # Only completed matches have real stats (incomplete rows are -1 / 0.00 placeholders)
    all_matches = all_matches[all_matches["status"] == "complete"].copy()

    # Parse date
    all_matches["date_GMT"] = pd.to_datetime(all_matches["date_GMT"], format="%b %d %Y - %I:%M%p")

    print(f"  Loaded {len(all_matches)} completed matches "
          f"from {len(filepaths)} file(s)")

    return all_matches


# ─── BUILD PER-TEAM ROLLING PROFILES ─────────────────────────────────────────

def build_team_xg_profiles(stats_df: pd.DataFrame, n_recent: int = 5, decay: float = 0.85) -> dict:
    """
    For each team, compute recency-weighted rolling stats from their last
    n_recent completed matches (across all loaded competitions):
      - xg_for      : actual xG generated (team_a_xg / team_b_xg)
      - xg_against  : actual xG conceded
      - goals_for / goals_against
      - ppg         : points per game (post-match, i.e. home_ppg/away_ppg AFTER this match)
      - possession
      - shots_on_target

    Returns {team_name: {stat: value, ...}}
    """
    # Reshape to one row per team per match (home and away both contribute)
    home = stats_df[[
        "date_GMT", "home_team_name", "team_a_xg", "team_b_xg",
        "home_team_goal_count", "away_team_goal_count",
        "home_ppg", "home_team_possession", "home_team_shots_on_target",
    ]].rename(columns={
        "home_team_name": "team",
        "team_a_xg": "xg_for", "team_b_xg": "xg_against",
        "home_team_goal_count": "goals_for", "away_team_goal_count": "goals_against",
        "home_ppg": "ppg", "home_team_possession": "possession",
        "home_team_shots_on_target": "shots_on_target",
    })

    away = stats_df[[
        "date_GMT", "away_team_name", "team_b_xg", "team_a_xg",
        "away_team_goal_count", "home_team_goal_count",
        "away_ppg", "away_team_possession", "away_team_shots_on_target",
    ]].rename(columns={
        "away_team_name": "team",
        "team_b_xg": "xg_for", "team_a_xg": "xg_against",
        "away_team_goal_count": "goals_for", "home_team_goal_count": "goals_against",
        "away_ppg": "ppg", "away_team_possession": "possession",
        "away_team_shots_on_target": "shots_on_target",
    })

    long_df = pd.concat([home, away], ignore_index=True)

    # Drop rows where xg is missing/zero placeholder (no real data)
    long_df = long_df[long_df["xg_for"] > 0].copy()

    profiles = {}
    for team, group in long_df.groupby("team"):
        recent = group.sort_values("date_GMT").tail(n_recent)
        if len(recent) == 0:
            continue

        n = len(recent)
        weights = np.array([decay ** (n - i - 1) for i in range(n)])
        weights = weights / weights.sum()

        profiles[team] = {
            "xg_for":          float(np.dot(recent["xg_for"],          weights)),
            "xg_against":      float(np.dot(recent["xg_against"],      weights)),
            "goals_for":       float(np.dot(recent["goals_for"],       weights)),
            "goals_against":   float(np.dot(recent["goals_against"],   weights)),
            "ppg":             float(np.dot(recent["ppg"],             weights)),
            "possession":      float(np.dot(recent["possession"],      weights)),
            "shots_on_target": float(np.dot(recent["shots_on_target"], weights)),
            "n_matches":       n,
        }

    print(f"  Built xG profiles for {len(profiles)} teams")
    return profiles


# ─── EXTRACT PRE-MATCH PREDICTIONS FOR UPCOMING WC MATCHES ───────────────────

def get_prematch_signals(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    For matches in the WC stats file (complete or not), extract the
    PRE-MATCH expectations: Pre-Match xG, Pre-Match PPG, and bookmaker odds
    embedded in the file (odds_ft_home_team_win etc).

    These represent footystats' own model + market — useful as another
    consensus source, exactly like load_consensus_odds().
    """
    cols_map = {
        "home_team_name": "team1",
        "away_team_name": "team2",
        "Home Team Pre-Match xG": "prematch_xg1",
        "Away Team Pre-Match xG": "prematch_xg2",
        "Pre-Match PPG (Home)":   "prematch_ppg1",
        "Pre-Match PPG (Away)":   "prematch_ppg2",
        "odds_ft_home_team_win":  "fs_odd_1_dec",
        "odds_ft_draw":           "fs_odd_null_dec",
        "odds_ft_away_team_win":  "fs_odd_2_dec",
    }
    out = stats_df[list(cols_map.keys())].rename(columns=cols_map).copy()

    # Convert decimal odds -> implied probability, then normalize (remove vig)
    for col, prob_col in [("fs_odd_1_dec", "fs_p1"), ("fs_odd_null_dec", "fs_pd"), ("fs_odd_2_dec", "fs_p2")]:
        out[prob_col] = np.where(out[col] > 0, 1 / out[col], np.nan)

    total = out[["fs_p1", "fs_pd", "fs_p2"]].sum(axis=1)
    for col in ["fs_p1", "fs_pd", "fs_p2"]:
        out[col] = out[col] / total

    return out[["team1", "team2",
                "prematch_xg1", "prematch_xg2",
                "prematch_ppg1", "prematch_ppg2",
                "fs_p1", "fs_pd", "fs_p2"]]


# ─── INTEGRATION INTO predict_match ──────────────────────────────────────────

def enrich_lambdas_with_xg(
    df_pred: pd.DataFrame,
    xg_profiles: dict,
    prematch: pd.DataFrame | None = None,
    xg_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Adjust lambda1/lambda2 using recent xG-based attack/defense strength,
    and optionally blend in footystats' pre-match xG for the *specific* upcoming match.

    xg_weight: how much weight the xG-derived lambda gets vs the existing
               model lambda (0.5 = equal blend).
    """
    df = df_pred.copy()

    # League-average xG per team per match (for normalization)
    all_xg_for = [p["xg_for"] for p in xg_profiles.values()]
    league_avg_xg = float(np.mean(all_xg_for)) if all_xg_for else 1.3

    df["xg_lambda1"] = np.nan
    df["xg_lambda2"] = np.nan
    df["lambda1_orig"] = df["lambda1"]
    df["lambda2_orig"] = df["lambda2"]

    for idx, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        p1 = xg_profiles.get(t1)
        p2 = xg_profiles.get(t2)

        if p1 is None or p2 is None:
            continue  # no recent data — keep original lambda

        # Attack/defense strength relative to league average
        attack1  = p1["xg_for"]     / league_avg_xg
        defense2 = p2["xg_against"] / league_avg_xg
        attack2  = p2["xg_for"]     / league_avg_xg
        defense1 = p1["xg_against"] / league_avg_xg

        xg_lambda1 = attack1 * defense2 * league_avg_xg
        xg_lambda2 = attack2 * defense1 * league_avg_xg

        df.at[idx, "xg_lambda1"] = round(xg_lambda1, 2)
        df.at[idx, "xg_lambda2"] = round(xg_lambda2, 2)

        # Blend with existing model lambda
        df.at[idx, "lambda1"] = (1 - xg_weight) * row["lambda1"] + xg_weight * xg_lambda1
        df.at[idx, "lambda2"] = (1 - xg_weight) * row["lambda2"] + xg_weight * xg_lambda2

    # Optionally blend in footystats' own pre-match xG for THIS specific match
    if prematch is not None:
        df = df.merge(
            prematch[["team1", "team2", "prematch_xg1", "prematch_xg2",
                     "prematch_ppg1", "prematch_ppg2", "fs_p1", "fs_pd", "fs_p2"]],
            on=["team1", "team2"], how="left",
        )

        has_prematch = df["prematch_xg1"].notna() & (df["prematch_xg1"] > 0)
        n_matched = has_prematch.sum()
        print(f"  Pre-match xG available for {n_matched}/{len(df)} matches")

        # Final blend: 50% rolling xG-adjusted lambda, 50% this match's specific pre-match xG
        df.loc[has_prematch, "lambda1"] = (
            0.5 * df.loc[has_prematch, "lambda1"] + 0.5 * df.loc[has_prematch, "prematch_xg1"]
        )
        df.loc[has_prematch, "lambda2"] = (
            0.5 * df.loc[has_prematch, "lambda2"] + 0.5 * df.loc[has_prematch, "prematch_xg2"]
        )

    df["lambda1"] = np.clip(df["lambda1"], 0.3, 4.0)
    df["lambda2"] = np.clip(df["lambda2"], 0.3, 4.0)

    # FIX: recompute probabilities and predicted score from the UPDATED lambdas.
    # p_win1/p_draw/p_win2/predicted_score were derived from the OLD lambdas
    # in predict_match() and are now stale — must be regenerated here.
    max_g = 7
    for idx, row in df.iterrows():
        l1, l2 = row["lambda1"], row["lambda2"]
        matrix = np.outer(
            [poisson.pmf(i, l1) for i in range(max_g)],
            [poisson.pmf(i, l2) for i in range(max_g)],
        )
        df.at[idx, "p_win1"] = round(float(np.tril(matrix, -1).sum()), 3)
        df.at[idx, "p_draw"] = round(float(np.trace(matrix)), 3)
        df.at[idx, "p_win2"] = round(float(np.triu(matrix,  1).sum()), 3)

        i, j = np.unravel_index(matrix.argmax(), matrix.shape)
        df.at[idx, "predicted_score"] = f"{i}-{j}"

    return df