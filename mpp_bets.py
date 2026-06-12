import pandas as pd
import numpy as np
from scipy.stats import poisson

# ─── MPP (MATCH POINTS PREDICTION) BET ANALYSIS ──────────────────────────────

def compute_mpp_bets(
    df_pred: pd.DataFrame,
    odds_mpp_path: str,
    max_g: int = 6,
) -> pd.DataFrame:
    """
    For each match, determine which SCORE to submit for the points-based
    prediction game (odds_mpp.csv: points awarded per outcome, not per exact score).

    Logic per match:
      1. Use df_pred's lambda1/lambda2 (your model's expected goals) to build
         the full Poisson scoreline grid.
      2. Group scorelines into 3 buckets: team1 win / draw / team2 win.
      3. For each bucket, pick the single most likely scoreline within it
         (e.g. if "team1 win" bucket, the most likely score where team1 > team2).
      4. EV for that bucket = P(bucket) * points_for_that_outcome.
      5. Recommend the bucket+score with highest EV — which may NOT be
         your model's single most-likely outcome overall (that's the "value bet").

    Returns a dataframe ready to save as mpp_bets.csv.
    """
    odds_mpp = pd.read_csv(odds_mpp_path)

    # Normalize pct columns (support 0-1 or 0-100)
    pct_cols = ["team1_win_pct", "draw_pct", "team2_win_pct"]
    for col in pct_cols:
        odds_mpp[col] = pd.to_numeric(odds_mpp[col], errors="coerce")
    if odds_mpp["team1_win_pct"].max() > 1.5:
        for col in pct_cols:
            odds_mpp[col] /= 100

    merged = df_pred.merge(
        odds_mpp[["team1", "team2"] + pct_cols +
                 ["team1_points", "draw_points", "team2_points"]],
        on=["team1", "team2"], how="left",
    )

    rows = []
    for _, r in merged.iterrows():
        l1, l2 = r["lambda1"], r["lambda2"]

        # Full Poisson grid
        grid = np.outer(
            [poisson.pmf(i, l1) for i in range(max_g + 1)],
            [poisson.pmf(j, l2) for j in range(max_g + 1)],
        )

        # Best scoreline PER OUTCOME BUCKET
        best = {}
        for outcome in ["team1_win", "draw", "team2_win"]:
            mask = np.zeros_like(grid, dtype=bool)
            for i in range(max_g + 1):
                for j in range(max_g + 1):
                    if outcome == "team1_win" and i > j:
                        mask[i, j] = True
                    elif outcome == "draw" and i == j:
                        mask[i, j] = True
                    elif outcome == "team2_win" and i < j:
                        mask[i, j] = True

            bucket_prob = grid[mask].sum()
            masked_grid = np.where(mask, grid, -1)
            i_best, j_best = np.unravel_index(masked_grid.argmax(), masked_grid.shape)

            best[outcome] = {
                "score": f"{i_best}-{j_best}",
                "p_outcome": bucket_prob,          # P(team1 wins / draw / team2 wins) overall
                "p_exact": grid[i_best, j_best],   # P(this specific scoreline)
            }

        # EV per outcome bucket = P(outcome) * points awarded for that outcome
        ev_team1 = best["team1_win"]["p_outcome"] * r["team1_points"]
        ev_draw  = best["draw"]["p_outcome"]      * r["draw_points"]
        ev_team2 = best["team2_win"]["p_outcome"] * r["team2_points"]

        evs = {"team1_win": ev_team1, "draw": ev_draw, "team2_win": ev_team2}
        best_outcome = max(evs, key=evs.get)

        # Platform's implied "expected" outcome (highest pct from odds_mpp)
        platform_probs = {
            "team1_win": r["team1_win_pct"],
            "draw":      r["draw_pct"],
            "team2_win": r["team2_win_pct"],
        }
        platform_favourite = max(platform_probs, key=platform_probs.get)

        # Model's most likely outcome overall (from p_win1/p_draw/p_win2 if present)
        model_probs = {
            "team1_win": r.get("p_win1", best["team1_win"]["p_outcome"]),
            "draw":      r.get("p_draw", best["draw"]["p_outcome"]),
            "team2_win": r.get("p_win2", best["team2_win"]["p_outcome"]),
        }
        model_favourite = max(model_probs, key=model_probs.get)

        # Edge = your model's prob for the RECOMMENDED outcome minus the
        # platform's prob for that SAME outcome (i.e. are you more confident
        # than the platform thinks you should be, in this specific outcome?)
        edge = model_probs[best_outcome] - platform_probs[best_outcome]

        # Naive EV: what you'd get by just always picking the platform favourite's score
        naive_score = best[platform_favourite]["score"]
        naive_ev    = platform_probs[platform_favourite] * \
                       {"team1_win": r["team1_points"], "draw": r["draw_points"],
                        "team2_win": r["team2_points"]}[platform_favourite]

        is_contrarian = (best_outcome != platform_favourite)

        # Risk classification
        if evs[best_outcome] < 30:
            risk = "LOW_RETURN"          # safe-ish but low points either way
        elif is_contrarian and edge > 0.05:
            risk = "VALUE_CONTRARIAN"    # your model disagrees with platform AND has edge
        elif is_contrarian:
            risk = "SPECULATIVE"         # contrarian but no clear edge — risky
        else:
            risk = "SAFE_FAVOURITE"      # agrees with platform favourite

        rows.append({
            "match_id":         r.get("match_id"),
            "date":             r.get("date"),
            "team1":            r["team1"],
            "team2":            r["team2"],
            "recommended_score": best[best_outcome]["score"],
            "recommended_outcome": best_outcome,
            "p_outcome":        round(best[best_outcome]["p_outcome"], 3),
            "p_exact_score":    round(best[best_outcome]["p_exact"], 3),
            "points_if_correct": {"team1_win": r["team1_points"], "draw": r["draw_points"],
                                  "team2_win": r["team2_points"]}[best_outcome],
            "expected_value":   round(evs[best_outcome], 2),
            "platform_favourite": platform_favourite,
            "platform_p":       round(platform_probs[platform_favourite], 3),
            "naive_score":      naive_score,
            "naive_ev":         round(naive_ev, 2),
            "ev_gain_vs_naive": round(evs[best_outcome] - naive_ev, 2),
            "edge_pct":         round(edge * 100, 1),
            "is_contrarian":    is_contrarian,
            "risk_level":       risk,
            # Reference: EV of all 3 buckets, for transparency
            "ev_team1_win":     round(ev_team1, 2),
            "ev_draw":          round(ev_draw, 2),
            "ev_team2_win":     round(ev_team2, 2),
            "score_team1_win":  best["team1_win"]["score"],
            "score_draw":       best["draw"]["score"],
            "score_team2_win":  best["team2_win"]["score"],
        })

    return pd.DataFrame(rows)


def print_mpp_summary(df: pd.DataFrame):
    """Quick console summary highlighting value bets."""
    n_contrarian = df["is_contrarian"].sum()
    n_value      = (df["risk_level"] == "VALUE_CONTRARIAN").sum()
    total_ev     = df["expected_value"].sum()
    naive_ev     = df["naive_ev"].sum()

    print("\n" + "="*78)
    print("MPP SCORE PREDICTIONS")
    print("="*78)
    print(f"  Matches            : {len(df)}")
    print(f"  Contrarian picks   : {n_contrarian}")
    print(f"  Value contrarians  : {n_value}  (edge > 5% AND disagrees with platform)")
    print(f"  Total EV           : {total_ev:.1f} pts")
    print(f"  Naive (favourite)  : {naive_ev:.1f} pts")
    print(f"  Gain vs naive      : {total_ev - naive_ev:+.1f} pts")

    value_bets = df[df["risk_level"] == "VALUE_CONTRARIAN"]
    if len(value_bets) > 0:
        print(f"\n{'─'*78}")
        print("  VALUE CONTRARIAN PICKS (worth double-checking)")
        print(f"{'─'*78}")
        for _, r in value_bets.iterrows():
            print(f"\n  {r['team1']} vs {r['team2']}")
            print(f"    Platform favours : {r['platform_favourite']} ({r['platform_p']:.1%})")
            print(f"    Recommend        : {r['recommended_score']} "
                  f"({r['recommended_outcome']}, edge {r['edge_pct']:+.1f}%)")
            print(f"    EV: {r['expected_value']:.1f} vs naive {r['naive_ev']:.1f} "
                  f"({r['ev_gain_vs_naive']:+.1f})")

    print(f"\n{'='*78}\n")