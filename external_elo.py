import pandas as pd
import numpy as np
from scipy.stats import poisson
from odds_integration import load_aliases, _fuzzy_match

# ─── LOAD EXTERNAL ELO / RANK DATA ────────────────────────────────────────────

def load_external_elo(filepath: str = "data/elo_ratings_wc2026.csv") -> dict:
    """
    Load pre-computed Elo ratings and ranks for 2026, indexed by country name.

    Returns {team_name: {"elo": float, "rank": int, "rank_avg": float, ...}}

    This replaces the self-computed Elo dict (built by replaying results.csv),
    since this snapshot already reflects full international match history
    through end of 2026 — more accurate than a 2010-onward replay.
    """
    df = pd.read_csv(filepath)

    elo_dict = {}
    for _, row in df.iterrows():
        elo_dict[row["country"]] = {
            "elo":          float(row["rating"]),
            "elo_max":      float(row["rating_max"]),
            "elo_avg":      float(row["rating_avg"]),
            "rank":         int(row["rank"]),
            "rank_avg":     float(row["rank_avg"]),
            "confederation": row["confederation"],
            "is_host":      bool(row["is_host"]),
            "matches_total": int(row["matches_total"]),
            "goal_diff_avg": (row["goals_for"] - row["goals_against"]) / max(row["matches_total"], 1),
        }

    print(f"  Loaded external Elo/rank for {len(elo_dict)} teams")
    return elo_dict


# ─── APPLY TO PREDICTIONS ─────────────────────────────────────────────────────

def apply_external_elo(
    df_pred: pd.DataFrame,
    elo_dict: dict,
    aliases_path: str = "data/team_aliases.csv",
) -> pd.DataFrame:
    """
    Overwrite elo1/elo2/rank1/rank2 in df_pred using the external 2026 source.
    Uses the same alias + fuzzy resolution as merge_odds() for team names.

    Adds a new 'elo_diff' column (elo1 - elo2) for convenience —
    useful if you want to re-weight lambdas based on this more accurate gap.
    """
    df = df_pred.copy()
    aliases = load_aliases(aliases_path)
    elo_teams = list(elo_dict.keys())

    def resolve(name: str) -> str | None:
        if name in elo_dict:
            return name
        alias_hit = aliases.get(name.strip().lower())
        if alias_hit and alias_hit in elo_dict:
            return alias_hit
        fuzzy_hit = _fuzzy_match(name, elo_teams)
        return fuzzy_hit

    # Save old values for comparison
    df["elo1_old"] = df["elo1"]
    df["elo2_old"] = df["elo2"]

    n_matched = 0
    for idx, row in df.iterrows():
        m1 = resolve(row["team1"])
        m2 = resolve(row["team2"])

        if m1 is None:
            print(f"  ⚠ No Elo match for '{row['team1']}' — keeping computed Elo")
        else:
            if m1 != row["team1"]:
                print(f"  ↳ Elo match: '{row['team1']}' -> '{m1}'")
            df.at[idx, "elo1"]  = elo_dict[m1]["elo"]
            df.at[idx, "rank1"] = elo_dict[m1]["rank"]

        if m2 is None:
            print(f"  ⚠ No Elo match for '{row['team2']}' — keeping computed Elo")
        else:
            if m2 != row["team2"]:
                print(f"  ↳ Elo match: '{row['team2']}' -> '{m2}'")
            df.at[idx, "elo2"]  = elo_dict[m2]["elo"]
            df.at[idx, "rank2"] = elo_dict[m2]["rank"]

        if m1 is not None and m2 is not None:
            n_matched += 1

    df["elo_diff"] = df["elo1"] - df["elo2"]

    print(f"  External Elo applied to {n_matched}/{len(df)} matches")
    return df


def reapply_elo_adjustment_to_lambdas(
    df_pred: pd.DataFrame,
    elo_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Re-apply the Elo-based lambda adjustment using the NEW elo_diff,
    replacing whatever adjustment was baked in with the old Elo.

    This mirrors the adjustment in predict_match():
        lambda *= (1 + elo_weight * clip(elo_diff/400, -0.5, 0.5))

    IMPORTANT: call this AFTER apply_external_elo() and BEFORE odds calibration,
    so the corrected Elo gap properly influences the lambdas that get calibrated.

    To avoid double-applying the old Elo adjustment, this function first
    REMOVES the old adjustment (using elo1_old/elo2_old) then applies the new one.
    """
    df = df_pred.copy()

    old_elo_adj = np.clip((df["elo1_old"] - df["elo2_old"]) / 400, -0.5, 0.5)
    new_elo_adj = np.clip((df["elo1"]     - df["elo2"])     / 400, -0.5, 0.5)

    # Remove old adjustment, apply new one
    df["lambda1"] = df["lambda1"] / (1 + elo_weight * old_elo_adj) * (1 + elo_weight * new_elo_adj)
    df["lambda2"] = df["lambda2"] / (1 - elo_weight * old_elo_adj) * (1 - elo_weight * new_elo_adj)

    df["lambda1"] = np.clip(df["lambda1"], 0.3, 4.0)
    df["lambda2"] = np.clip(df["lambda2"], 0.3, 4.0)

    # FIX: recompute p_win1/p_draw/p_win2/predicted_score from updated lambdas
    # — same stale-value issue as enrich_lambdas_with_xg().
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