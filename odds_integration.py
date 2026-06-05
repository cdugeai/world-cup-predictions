import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

# ─── ODDS UTILITIES ──────────────────────────────────────────────────────────

def fractional_to_prob(odd_str: str) -> float:
    """'33/20' -> raw implied probability (before margin removal)"""
    odd_str = str(odd_str).strip()
    if "/" not in odd_str:
        raise ValueError(f"Expected fractional odd like '33/20', got: {odd_str!r}")
    num, den = map(int, odd_str.split("/"))
    return den / (num + den)

def normalize_probs(p1: float, pd: float, p2: float) -> tuple[float, float, float]:
    """Remove bookmaker margin (vig) so probabilities sum exactly to 1.0"""
    total = p1 + pd + p2
    return p1 / total, pd / total, p2 / total

def load_odds(filepath: str) -> pd.DataFrame:
    """Load odds CSV and compute normalized market probabilities."""
    odds = pd.read_csv(filepath)

    odds["p1_raw"] = odds["odd_1"].apply(fractional_to_prob)
    odds["pd_raw"] = odds["odd_null"].apply(fractional_to_prob)
    odds["p2_raw"] = odds["odd_2"].apply(fractional_to_prob)

    # Bookmaker margin: how much over 100% the raw probs sum to
    odds["margin"] = odds["p1_raw"] + odds["pd_raw"] + odds["p2_raw"] - 1.0

    normalized = odds.apply(
        lambda r: normalize_probs(r["p1_raw"], r["pd_raw"], r["p2_raw"]), axis=1
    )
    odds[["mkt_win1", "mkt_draw", "mkt_win2"]] = pd.DataFrame(
        normalized.tolist(), index=odds.index
    )

    return odds[["team1", "team2", "odd_1", "odd_null", "odd_2",
                 "mkt_win1", "mkt_draw", "mkt_win2", "margin"]]


# ─── LAMBDA CALIBRATION ──────────────────────────────────────────────────────

def poisson_probs(l1: float, l2: float, max_g: int = 7) -> tuple[float, float, float]:
    """Compute win/draw/win probabilities from two Poisson lambdas."""
    matrix = np.outer(
        [poisson.pmf(i, l1) for i in range(max_g)],
        [poisson.pmf(i, l2) for i in range(max_g)],
    )
    return (
        float(np.tril(matrix, -1).sum()),  # team1 win
        float(np.trace(matrix)),            # draw
        float(np.triu(matrix,  1).sum()),   # team2 win
    )

def calibrate_lambdas(
    p1_target: float,
    pd_target: float,
    p2_target: float,
    l1_init: float,
    l2_init: float,
) -> tuple[float, float]:
    """
    Find lambdas whose Poisson distribution best matches the market probabilities.
    Uses the model's own lambdas as starting point so calibration stays realistic.
    """
    def loss(params):
        l1, l2 = params
        if l1 <= 0.1 or l2 <= 0.1:
            return 1e6
        pw1, pd, pw2 = poisson_probs(l1, l2)
        return (pw1 - p1_target)**2 + (pd - pd_target)**2 + (pw2 - p2_target)**2

    result = minimize(
        loss,
        x0=[l1_init, l2_init],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    l1_cal, l2_cal = result.x
    return float(np.clip(l1_cal, 0.3, 4.0)), float(np.clip(l2_cal, 0.3, 4.0))


# ─── MERGE ODDS INTO PREDICTIONS ─────────────────────────────────────────────

def merge_odds(df_pred: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Join odds to predictions on team names (case-insensitive, stripped).
    Unmatched rows keep model-only probabilities.
    """
    df_pred = df_pred.copy()
    df_odds  = df_odds.copy()

    # Normalize team names for joining
    for df in [df_pred, df_odds]:
        for col in ["team1", "team2"]:
            df[col] = df[col].str.strip().str.lower()

    merged = df_pred.merge(
        df_odds[["team1", "team2", "odd_1", "odd_null", "odd_2",
                 "mkt_win1", "mkt_draw", "mkt_win2", "margin"]],
        on=["team1", "team2"],
        how="left",
    )

    # Restore original casing from predictions
    merged["team1"] = df_pred["team1"].values
    merged["team2"] = df_pred["team2"].values

    return merged


def apply_odds_calibration(
    df: pd.DataFrame,
    market_weight: float = 0.7,
) -> pd.DataFrame:
    """
    For each row that has odds:
      1. Blend model + market probabilities
      2. Calibrate Poisson lambdas to match blended probs
      3. Re-derive predicted score from calibrated lambdas
      4. Add diagnostic columns showing model vs market disagreement
    """
    df = df.copy()

    # New columns
    df["p_win1_market"]  = np.nan
    df["p_draw_market"]  = np.nan
    df["p_win2_market"]  = np.nan
    df["p_win1_final"]   = df["p_win1"]
    df["p_draw_final"]   = df["p_draw"]
    df["p_win2_final"]   = df["p_win2"]
    df["lambda1_cal"]    = df["lambda1"]
    df["lambda2_cal"]    = df["lambda2"]
    df["score_cal"]      = df["predicted_score"]
    df["has_odds"]       = False
    df["model_vs_market"]= ""  # human-readable disagreement flag

    MODEL_W = 1 - market_weight

    for idx, row in df.iterrows():
        if pd.isna(row.get("mkt_win1")):
            continue  # no odds for this match

        df.at[idx, "has_odds"] = True
        mw1, mwd, mw2 = row["mkt_win1"], row["mkt_draw"], row["mkt_win2"]
        df.at[idx, "p_win1_market"] = round(mw1, 3)
        df.at[idx, "p_draw_market"] = round(mwd, 3)
        df.at[idx, "p_win2_market"] = round(mw2, 3)

        # Blended probabilities
        pw1 = market_weight * mw1 + MODEL_W * row["p_win1"]
        pwd = market_weight * mwd + MODEL_W * row["p_draw"]
        pw2 = market_weight * mw2 + MODEL_W * row["p_win2"]

        # Re-normalize (floating point safety)
        total = pw1 + pwd + pw2
        pw1, pwd, pw2 = pw1/total, pwd/total, pw2/total

        df.at[idx, "p_win1_final"] = round(pw1, 3)
        df.at[idx, "p_draw_final"] = round(pwd, 3)
        df.at[idx, "p_win2_final"] = round(pw2, 3)

        # Calibrate lambdas to blended probs
        l1_cal, l2_cal = calibrate_lambdas(
            pw1, pwd, pw2,
            l1_init=row["lambda1"],
            l2_init=row["lambda2"],
        )
        df.at[idx, "lambda1_cal"] = round(l1_cal, 2)
        df.at[idx, "lambda2_cal"] = round(l2_cal, 2)

        # New predicted score from calibrated lambdas
        max_g = 7
        matrix = np.outer(
            [poisson.pmf(i, l1_cal) for i in range(max_g)],
            [poisson.pmf(i, l2_cal) for i in range(max_g)],
        )
        i, j = np.unravel_index(matrix.argmax(), matrix.shape)
        df.at[idx, "score_cal"] = f"{i}-{j}"

        # Flag big model vs market disagreements (useful for analysis)
        diff = row["p_win1"] - mw1
        if abs(diff) > 0.15:
            direction = "model favours team1" if diff > 0 else "market favours team1"
            df.at[idx, "model_vs_market"] = f"⚠ {direction} ({abs(diff):.0%} gap)"

    return df


# ─── DISPLAY ─────────────────────────────────────────────────────────────────

def print_predictions(df: pd.DataFrame):
    print("\n" + "="*80)
    print("MATCH PREDICTIONS")
    print("="*80)

    for _, r in df.iterrows():
        print(f"\n  {r['team1']} vs {r['team2']}  [{r.get('group','')} — {r.get('date','')}]")
        print(f"  {'─'*50}")

        if r.get("has_odds"):
            print(f"  Odds:       {r['odd_1']} / {r['odd_null']} / {r['odd_2']}"
                  f"  (margin: {r['margin']:.1%})")
            print(f"  Market:     W1={r['p_win1_market']:.1%}  D={r['p_draw_market']:.1%}"
                  f"  W2={r['p_win2_market']:.1%}")
            print(f"  Model:      W1={r['p_win1']:.1%}  D={r['p_draw']:.1%}  W2={r['p_win2']:.1%}")
            print(f"  ── Final (blended) ──────────────────────────────")
            print(f"  Probs:      W1={r['p_win1_final']:.1%}  D={r['p_draw_final']:.1%}"
                  f"  W2={r['p_win2_final']:.1%}")
            print(f"  Score:      {r['score_cal']}  "
                  f"(λ1={r['lambda1_cal']}  λ2={r['lambda2_cal']})")
            if r["model_vs_market"]:
                print(f"  {r['model_vs_market']}")
        else:
            print(f"  [no odds — model only]")
            print(f"  Probs:      W1={r['p_win1']:.1%}  D={r['p_draw']:.1%}  W2={r['p_win2']:.1%}")
            print(f"  Score:      {r['predicted_score']}  "
                  f"(λ1={r['lambda1']}  λ2={r['lambda2']})")

    print("\n" + "="*80)