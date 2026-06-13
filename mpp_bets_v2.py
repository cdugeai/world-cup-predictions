#!/usr/bin/env python3
"""
mpp_bets_v2.py

Generates score/outcome recommendations for a "pronostic" (correct-score
prediction) game, maximising expected points given the platform's scoring
rules (team1_points, draw_points, team2_points awarded for guessing the
right RESULT — win/draw/win — for a given match).

Pipeline
--------
1. Load all CSV sources, normalise team names via team_aliases.csv.
2. For each match in matches_group.csv, gather:
     - platform scoring odds (odds_mpp.csv) -> points for each outcome
     - bookmaker market odds (odds_unibet.csv, fallback odds_checker_com.csv)
       -> implied probabilities (de-vigged) = "platform_favourite/platform_p"
     - Elo ratings (elo_ratings_wc2026.csv) -> Elo-based win/draw/loss probs
     - historical match stats (friendlies + WC matches) -> goals/xG priors
       for a Poisson scoreline model
3. Blend Elo probs and de-vigged market probs into a single p(outcome)
   estimate (model probability), then build a Poisson scoreline grid using
   each side's expected goals (lambda) to get p_exact_score per scoreline
   and the most likely scoreline per outcome (score_team1_win/score_draw/
   score_team2_win).
4. For every outcome, compute EV = p(outcome) * points(outcome).
   ev_team1_win, ev_draw, ev_team2_win are these three values.
   The recommended_outcome is the argmax; recommended_score is the most
   likely scoreline consistent with that outcome (from the Poisson grid).
5. naive_score / naive_ev: the "obvious" pick = bet on the platform
   favourite (highest platform_p) with the most common scoreline for that
   outcome (1-0 / 1-1 / 0-1), and its EV under the model probabilities.
6. ev_gain_vs_naive = expected_value - naive_ev
   edge_pct = (model_p_for_chosen_outcome - platform_p_for_chosen_outcome)
              * 100   -> how much our model's probability for the picked
              outcome diverges from the bookmaker-implied probability for
              the SAME outcome as picked by the model.
   is_contrarian = recommended_outcome != platform_favourite
   risk_level:
       FOLLOW_FAVOURITE  -> recommended_outcome == platform_favourite
       VALUE_CONTRARIAN  -> is_contrarian AND edge_pct > 0 (positive edge)
       SPECULATIVE       -> is_contrarian AND edge_pct <= 0
7. If actual results are available (filled in international-world-cup-
   matches CSV, status == "complete"), actual_score / actual_outcome /
   outcome_correct / score_correct are computed; otherwise left blank.

Run:
    python3 mpp_bets_v2.py --data-dir ./data --out mpp_bets_v2_output.csv
"""

import argparse
import csv
import math
import os
import re
from fractions import Fraction
import polars as pl


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_alias_map(rows):
    """alias -> canonical, plus identity mapping (case-insensitive)."""
    m = {}
    for r in rows:
        alias = r["alias"].strip()
        canon = r["canonical"].strip()
        m[alias.lower()] = canon
        m[canon.lower()] = canon
    return m


def canon(name, alias_map):
    name = name.strip()
    return alias_map.get(name.lower(), name)


def frac_odds_to_decimal(s):
    """Convert 'a/b' fractional odds to decimal odds (a/b + 1)."""
    s = s.strip()
    if not s:
        return None
    try:
        if "/" in s:
            f = Fraction(s)
            return float(f) + 1.0
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def implied_probs_from_decimal(d1, dx, d2):
    """De-vig three decimal odds into probabilities summing to 1."""
    raw = []
    for d in (d1, dx, d2):
        raw.append(1.0 / d if d and d > 0 else 0.0)
    total = sum(raw)
    if total == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return tuple(r / total for r in raw)


def elo_win_draw_loss(elo_a, elo_b, draw_factor=0.28):
    """
    Standard Elo win expectation, with a draw probability heuristic.
    draw_factor controls how much probability mass goes to the draw,
    peaking when the teams are evenly matched.
    """
    diff = elo_a - elo_b
    p_a_vs_b = 1.0 / (1.0 + 10 ** (-diff / 400.0))  # raw win prob for A (no draw)

    # Draw probability: higher when teams are closely matched.
    closeness = 1.0 - abs(p_a_vs_b - 0.5) * 2.0  # 1 when p=0.5, 0 when p=0 or 1
    p_draw = draw_factor * (0.5 + 0.5 * closeness) * 0.5  # scale into a sane range

    remaining = 1.0 - p_draw
    p_a = p_a_vs_b * remaining
    p_b = (1.0 - p_a_vs_b) * remaining
    return p_a, p_draw, p_b


def poisson_pmf(k, lam):
    if lam <= 0:
        lam = 0.05
    return math.exp(-lam) * lam ** k / math.factorial(k)


def estimate_lambdas(elo_a, elo_b, base_goals=1.35):
    """
    Crude expected-goals estimator: split a baseline total around the
    Elo-implied strength ratio. Clamped to sane bounds.
    """
    diff = elo_a - elo_b
    strength_ratio = 1.0 / (1.0 + 10 ** (-diff / 400.0))  # 0..1, >0.5 favours A
    total_goals = base_goals * 2
    lam_a = total_goals * strength_ratio
    lam_b = total_goals * (1 - strength_ratio)
    lam_a = max(0.3, min(3.5, lam_a))
    lam_b = max(0.3, min(3.5, lam_b))
    return lam_a, lam_b


def scoreline_grid(lam_a, lam_b, max_goals=6):
    """Return dict {(a,b): probability} over a Poisson scoreline grid."""
    grid = {}
    for a in range(max_goals + 1):
        for b in range(max_goals + 1):
            grid[(a, b)] = poisson_pmf(a, lam_a) * poisson_pmf(b, lam_b)
    # Normalise (tail beyond max_goals is negligible but tidy up anyway)
    total = sum(grid.values())
    if total > 0:
        grid = {k: v / total for k, v in grid.items()}
    return grid


def best_scoreline_for_outcome(grid, outcome):
    """Most probable scoreline (a,b) matching the outcome."""
    best, best_p = None, -1.0
    for (a, b), p in grid.items():
        if outcome == "team1_win" and a > b:
            ok = True
        elif outcome == "draw" and a == b:
            ok = True
        elif outcome == "team2_win" and a < b:
            ok = True
        else:
            ok = False
        if ok and p > best_p:
            best, best_p = (a, b), p
    return best, best_p


def fmt_score(t):
    return f"{t[0]}-{t[1]}"


def blend(model_p, market_p, weight_model=0.6):
    """Weighted blend of Elo-model probabilities and market-implied probabilities."""
    weight_market = 1.0 - weight_model
    blended = [m * weight_model + k * weight_market for m, k in zip(model_p, market_p)]
    total = sum(blended)
    return tuple(b / total for b in blended)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="mpp_bets_v2_output.csv")
    ap.add_argument("--weight-model", type=float, default=0.6,
                     help="Weight given to Elo-derived probabilities vs market odds (0-1)")
    args = ap.parse_args()

    D = args.data_dir

    aliases = build_alias_map(load_csv(os.path.join(D, "cleaned/team_aliases.csv")))
    odds_mpp = load_csv(os.path.join(D, "cleaned/odds_mpp.csv"))
    elo_rows = load_csv(os.path.join(D, "elo_ratings_wc2026.csv"))
    matches = load_csv(os.path.join(D, "cleaned/matchs_group_clean.csv"))
    unibet = load_csv(os.path.join(D, "cleaned/odds_unibet.csv"))
    checker = load_csv(os.path.join(D, "cleaned/odds_checker_com.csv"))
    wc_results = load_csv(os.path.join(D, "international-world-cup-matches-2026-to-2026-stats.csv"))

    # --- Index helpers -----------------------------------------------------
    elo_by_team = {}
    for r in elo_rows:
        elo_by_team[canon(r["country"], aliases)] = float(r["rating"])

    def get_elo(team):
        c = canon(team, aliases)
        return elo_by_team.get(c, 1500.0)  # default average rating if missing

    # platform scoring points, keyed by (team1, team2) canonical pair
    mpp_by_pair = {}
    for r in odds_mpp:
        t1, t2 = canon(r["team1"], aliases), canon(r["team2"], aliases)
        mpp_by_pair[(t1, t2)] = {
            "team1_points": int(r["team1_points"]),
            "draw_points": int(r["draw_points"]),
            "team2_points": int(r["team2_points"]),
        }

    # bookmaker odds: try unibet first, fallback to odds_checker
    def market_odds_for(t1, t2):
        for source in (unibet, checker):
            for r in source:
                rt1, rt2 = canon(r["team1"], aliases), canon(r["team2"], aliases)
                if rt1 == t1 and rt2 == t2:
                    d1 = frac_odds_to_decimal(r["odd_1"])
                    dx = frac_odds_to_decimal(r["odd_null"])
                    d2 = frac_odds_to_decimal(r["odd_2"])
                    if d1 and dx and d2:
                        return d1, dx, d2
        return None

    # completed-match results, keyed by canonical (home, away)
    actuals = {}
    for r in wc_results:
        if r.get("status") != "complete":
            continue
        home, away = canon(r["home_team_name"], aliases), canon(r["away_team_name"], aliases)
        try:
            hg, ag = int(r["home_team_goal_count"]), int(r["away_team_goal_count"])
        except (ValueError, KeyError):
            continue
        actuals[(home, away)] = (hg, ag)

    # --- Build output rows ---------------------------------------------------
    out_rows = []
    for m in matches:
        match_id = m["match_id"]
        date = m["date_"]
        t1, t2 = canon(m["team1"], aliases), canon(m["team2"], aliases)

        pts = mpp_by_pair.get((t1, t2))
        if pts is None:
            # try odds_mpp using reversed pair as fallback (shouldn't normally happen)
            pts = mpp_by_pair.get((t2, t1))
            if pts is None:
                continue  # no scoring info for this match -> skip

        team1_points = pts["team1_points"]
        draw_points = pts["draw_points"]
        team2_points = pts["team2_points"]

        # --- Elo model probabilities ---
        elo1, elo2 = get_elo(t1), get_elo(t2)
        elo_p1, elo_pd, elo_p2 = elo_win_draw_loss(elo1, elo2)

        # --- Market probabilities ---
        mkt = market_odds_for(t1, t2)
        if mkt:
            d1, dx, d2 = mkt
            mkt_p1, mkt_pd, mkt_p2 = implied_probs_from_decimal(d1, dx, d2)
        else:
            mkt_p1, mkt_pd, mkt_p2 = elo_p1, elo_pd, elo_p2  # fallback: same as model

        # --- Blended model probability ---
        p1, pd_, p2 = blend((elo_p1, elo_pd, elo_p2), (mkt_p1, mkt_pd, mkt_p2),
                             weight_model=args.weight_model)

        # --- Expected goals & scoreline grid ---
        lam1, lam2 = estimate_lambdas(elo1, elo2)
        grid = scoreline_grid(lam1, lam2)

        # rescale grid so that win/draw/loss marginal matches blended p1/pd_/p2
        marg_p1 = sum(p for (a, b), p in grid.items() if a > b)
        marg_pd = sum(p for (a, b), p in grid.items() if a == b)
        marg_p2 = sum(p for (a, b), p in grid.items() if a < b)

        adj_grid = {}
        for (a, b), p in grid.items():
            if a > b and marg_p1 > 0:
                adj_grid[(a, b)] = p * (p1 / marg_p1)
            elif a == b and marg_pd > 0:
                adj_grid[(a, b)] = p * (pd_ / marg_pd)
            elif a < b and marg_p2 > 0:
                adj_grid[(a, b)] = p * (p2 / marg_p2)
            else:
                adj_grid[(a, b)] = p
        total = sum(adj_grid.values())
        if total > 0:
            adj_grid = {k: v / total for k, v in adj_grid.items()}

        # --- Best scoreline per outcome ---
        score_t1, p_score_t1 = best_scoreline_for_outcome(adj_grid, "team1_win")
        score_d, p_score_d = best_scoreline_for_outcome(adj_grid, "draw")
        score_t2, p_score_t2 = best_scoreline_for_outcome(adj_grid, "team2_win")

        # --- EV per outcome ---
        ev_t1 = p1 * team1_points
        ev_d = pd_ * draw_points
        ev_t2 = p2 * team2_points

        ev_map = {
            "team1_win": (ev_t1, score_t1, p1, p_score_t1),
            "draw": (ev_d, score_d, pd_, p_score_d),
            "team2_win": (ev_t2, score_t2, p2, p_score_t2),
        }
        recommended_outcome = max(ev_map, key=lambda k: ev_map[k][0])
        expected_value, rec_score_tuple, p_outcome, p_exact_score = ev_map[recommended_outcome]
        recommended_score = fmt_score(rec_score_tuple) if rec_score_tuple else "0-0"
        points_if_correct = {"team1_win": team1_points,
                              "draw": draw_points,
                              "team2_win": team2_points}[recommended_outcome]

        # --- Platform favourite (by market-implied probability) ---
        market_probs = {"team1_win": mkt_p1, "draw": mkt_pd, "team2_win": mkt_p2}
        platform_favourite = max(market_probs, key=lambda k: market_probs[k])
        platform_p = round(market_probs[platform_favourite], 2)

        # --- Naive strategy: bet on platform favourite, typical scoreline ---
        typical_score = {"team1_win": (1, 0), "draw": (1, 1), "team2_win": (0, 1)}
        naive_score_tuple = typical_score[platform_favourite]
        naive_score = fmt_score(naive_score_tuple)
        naive_points = {"team1_win": team1_points,
                         "draw": draw_points,
                         "team2_win": team2_points}[platform_favourite]
        naive_p_outcome = {"team1_win": p1, "draw": pd_, "team2_win": p2}[platform_favourite]
        naive_ev = naive_p_outcome * naive_points

        ev_gain_vs_naive = expected_value - naive_ev

        # --- Edge & risk classification ---
        edge_pct = (p_outcome - market_probs[recommended_outcome]) * 100.0
        is_contrarian = recommended_outcome != platform_favourite
        if not is_contrarian:
            risk_level = "FOLLOW_FAVOURITE"
        elif edge_pct > 0:
            risk_level = "VALUE_CONTRARIAN"
        else:
            risk_level = "SPECULATIVE"

        # --- Actuals (if available) ---
        actual = actuals.get((t1, t2))
        if actual:
            ag1, ag2 = actual
            actual_score = fmt_score((ag1, ag2))
            if ag1 > ag2:
                actual_outcome = "team1_win"
            elif ag1 == ag2:
                actual_outcome = "draw"
            else:
                actual_outcome = "team2_win"
            outcome_correct = (actual_outcome == recommended_outcome)
            score_correct = (actual_score == recommended_score)
        else:
            actual_score = ""
            actual_outcome = ""
            outcome_correct = ""
            score_correct = ""

        out_rows.append({
            "match_id": match_id,
            "date": date,
            "team1": t1,
            "team2": t2,
            "recommended_score": recommended_score,
            "recommended_outcome": recommended_outcome,
            "p_outcome": round(p_outcome, 3),
            "p_exact_score": round(p_exact_score, 3),
            "points_if_correct": points_if_correct,
            "expected_value": round(expected_value, 2),
            "platform_favourite": platform_favourite,
            "platform_p": platform_p,
            "naive_score": naive_score,
            "naive_ev": round(naive_ev, 2),
            "ev_gain_vs_naive": round(ev_gain_vs_naive, 2),
            "edge_pct": round(edge_pct, 1),
            "is_contrarian": is_contrarian,
            "risk_level": risk_level,
            "ev_team1_win": round(ev_t1, 2),
            "ev_draw": round(ev_d, 2),
            "ev_team2_win": round(ev_t2, 2),
            "score_team1_win": fmt_score(score_t1) if score_t1 else "",
            "score_draw": fmt_score(score_d) if score_d else "",
            "score_team2_win": fmt_score(score_t2) if score_t2 else "",
            "actual_score": actual_score,
            "actual_outcome": actual_outcome,
            "outcome_correct": outcome_correct,
            "score_correct": score_correct,
        })

    # --- Write output CSV ---
    fieldnames = [
        "match_id", "date", "team1", "team2", "recommended_score", "recommended_outcome",
        "p_outcome", "p_exact_score", "points_if_correct", "expected_value",
        "platform_favourite", "platform_p", "naive_score", "naive_ev",
        "ev_gain_vs_naive", "edge_pct", "is_contrarian", "risk_level",
        "ev_team1_win", "ev_draw", "ev_team2_win",
        "score_team1_win", "score_draw", "score_team2_win",
        "actual_score", "actual_outcome", "outcome_correct", "score_correct",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"Wrote {len(out_rows)} rows to {args.out}")
    # Clean version
    clean_file_path = args.out.replace(".csv", "_clean.csv")
    (
        pl.read_csv(args.out)
        .select(
            "match_id", "date", "team1", "team2", "recommended_score", "is_contrarian", "risk_level", "expected_value", "naive_score", "naive_ev", "ev_gain_vs_naive", "edge_pct",
            "actual_score","actual_outcome","outcome_correct","score_correct"
        )
        .write_csv(clean_file_path)
    )
    print(f"Wrote {len(out_rows)} rows to {clean_file_path}")



if __name__ == "__main__":
    main()

# uv run python3 mpp_bets_v2.py --data-dir ./data --out data/out/mpp_bets_v2.csv