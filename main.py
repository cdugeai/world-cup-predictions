import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import poisson
from odds_integration import load_odds, merge_odds, apply_odds_calibration, print_predictions

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────

results   = pd.read_csv("data/results.csv", parse_dates=["date"])
rankings  = pd.read_csv("data/fifa_mens_rank.csv")
matches   = pd.read_csv("data/cleaned/matchs_group_clean.csv")

# FIX 1: Force numeric types on score columns (they can be read as strings)
results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
results = results.dropna(subset=["home_score", "away_score"])

# Fix team name mismatches between datasets — run Q1 check to find more
name_map = {}
# Load alias mapping from CSV
for x in pl.read_csv('data/cleaned/team_aliases.csv').to_dicts():
    name_map[x['alias']] = x['canonical']
matches["team1"] = matches["team1"].replace(name_map)
matches["team2"] = matches["team2"].replace(name_map)

# Focus on post-2010 matches only
results = results[results["date"] >= "2010-01-01"].copy()

# ─── 2. ELO RATINGS ──────────────────────────────────────────────────────────

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def update_elo(r_a, r_b, result, k=32):
    exp = expected_score(r_a, r_b)
    return r_a + k * (result - exp), r_b + k * ((1 - result) - (1 - exp))

K_FACTORS = {
    "FIFA World Cup":  60,
    "UEFA Euro":       50,
    "Copa América":    50,
    "AFC Asian Cup":   45,
    "Friendly":        20,
}
DEFAULT_K = 35

elo = {}
for _, row in results.sort_values("date").iterrows():
    h, a = row["home_team"], row["away_team"]
    r_h = elo.get(h, 1500)
    r_a = elo.get(a, 1500)
    if row["home_score"] > row["away_score"]:
        result = 1.0
    elif row["home_score"] == row["away_score"]:
        result = 0.5
    else:
        result = 0.0
    k = K_FACTORS.get(row["tournament"], DEFAULT_K)
    elo[h], elo[a] = update_elo(r_h, r_a, result, k)

# ─── 3. RECENT FORM ──────────────────────────────────────────────────────────

def team_form(team, results_df, n=10, decay=0.85):
    team_matches = results_df[
        (results_df["home_team"] == team) | (results_df["away_team"] == team)
    ].sort_values("date").tail(n)

    if len(team_matches) == 0:
        return {"form": 0.5, "avg_scored": 1.2, "avg_conceded": 1.2}

    outcomes, scored, conceded = [], [], []
    for _, r in team_matches.iterrows():
        if r["home_team"] == team:
            s, c = float(r["home_score"]), float(r["away_score"])  # FIX 2: explicit float cast
        else:
            s, c = float(r["away_score"]), float(r["home_score"])
        outcomes.append(1.0 if s > c else (0.5 if s == c else 0.0))
        scored.append(s)
        conceded.append(c)

    n_matches = len(outcomes)
    weights = np.array([decay ** (n_matches - i - 1) for i in range(n_matches)])
    return {
        "form":         float(np.average(outcomes,  weights=weights)),
        "avg_scored":   float(np.average(scored,    weights=weights)),
        "avg_conceded": float(np.average(conceded,  weights=weights)),
    }

# ─── 4. FIFA RANKING LOOKUP ───────────────────────────────────────────────────

latest_rankings = (
    rankings.sort_values(["team", "date", "semester"])
    .groupby("team").last().reset_index()
    [["team", "rank", "total.points"]]
)
ranking_dict = dict(zip(latest_rankings["team"], latest_rankings["rank"]))

# ─── 5. LEAGUE AVERAGES (for Dixon-Coles style attack/defense) ───────────────

# Compute global average goals per team per WC match (used to normalize)
wc = results[results["tournament"] == "FIFA World Cup"].copy()
if len(wc) == 0:
    # Fallback to all competitions
    wc = results.copy()

all_goals = pd.concat([
    wc[["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "conceded"}),
    wc[["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "conceded"}),
])
LEAGUE_AVG = float(all_goals["scored"].mean())  # ~1.1–1.3 for WC

print(f"League average goals per team per match: {LEAGUE_AVG:.3f}")

# Compute per-team attack/defense strengths relative to league average
team_stats = all_goals.groupby("team").agg(
    avg_scored   = ("scored",   "mean"),
    avg_conceded = ("conceded", "mean"),
    n_matches    = ("scored",   "count"),
).reset_index()

# Attack strength = team avg scored / league avg
# Defense weakness = team avg conceded / league avg  (higher = worse defense)
team_stats["attack"]  = team_stats["avg_scored"]   / LEAGUE_AVG
team_stats["defense"] = team_stats["avg_conceded"]  / LEAGUE_AVG

stats_dict = team_stats.set_index("team").to_dict("index")

def get_stats(team):
    """Return stats with fallback for unknown teams."""
    if team in stats_dict:
        return stats_dict[team]
    return {"attack": 1.0, "defense": 1.0, "avg_scored": LEAGUE_AVG, "avg_conceded": LEAGUE_AVG}

# ─── 6. PREDICT A MATCH ──────────────────────────────────────────────────────
ELO_WEIGHT = 0.15

def predict_match(team1, team2, results_df):
    # Recent form (exponentially weighted)
    f1 = team_form(team1, results_df)
    f2 = team_form(team2, results_df)

    elo1 = elo.get(team1, 1500)
    elo2 = elo.get(team2, 1500)

    # FIX 3: Correct Dixon-Coles style expected goals
    # lambda = attack_strength(team) * defense_weakness(opponent) * league_avg
    # Blend historical stats (60%) with recent form (40%) for robustness
    s1_hist = get_stats(team1)
    s2_hist = get_stats(team2)

    attack1  = 0.6 * s1_hist["attack"]  + 0.4 * (f1["avg_scored"]   / LEAGUE_AVG)
    defense2 = 0.6 * s2_hist["defense"] + 0.4 * (f2["avg_conceded"]  / LEAGUE_AVG)
    attack2  = 0.6 * s2_hist["attack"]  + 0.4 * (f2["avg_scored"]   / LEAGUE_AVG)
    defense1 = 0.6 * s1_hist["defense"] + 0.4 * (f1["avg_conceded"]  / LEAGUE_AVG)

    lambda1 = attack1 * defense2 * LEAGUE_AVG
    lambda2 = attack2 * defense1 * LEAGUE_AVG

    # Apply Elo-based adjustment (±15% max)
    elo_adj = np.clip((elo1 - elo2) / 400, -0.5, 0.5)
    lambda1 *= (1 + ELO_WEIGHT * elo_adj)
    lambda2 *= (1 - ELO_WEIGHT * elo_adj)

    # Clamp to sane range
    lambda1 = float(np.clip(lambda1, 0.3, 4.0))
    lambda2 = float(np.clip(lambda2, 0.3, 4.0))

    # Poisson probability matrix (scores 0–6)
    max_g = 7
    prob_matrix = np.outer(
        [poisson.pmf(i, lambda1) for i in range(max_g)],
        [poisson.pmf(i, lambda2) for i in range(max_g)],
    )

    p_win1 = float(np.tril(prob_matrix, -1).sum())
    p_draw = float(np.trace(prob_matrix))
    p_win2 = float(np.triu(prob_matrix,  1).sum())

    idx = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)

    return {
        "team1": team1, "team2": team2,
        "lambda1":        round(lambda1, 2),
        "lambda2":        round(lambda2, 2),
        "p_win1":         round(p_win1, 3),
        "p_draw":         round(p_draw, 3),
        "p_win2":         round(p_win2, 3),
        "predicted_score": f"{idx[0]}-{idx[1]}",
        "elo1":  round(elo1), "elo2": round(elo2),
        "rank1": ranking_dict.get(team1, "N/A"),
        "rank2": ranking_dict.get(team2, "N/A"),
    }


# ─── Utils: Compare prediction with actual results ────────────────────────────────────────
from scoring import load_actual_results, add_results
actual = load_actual_results("data/international-world-cup-matches-2026-to-2026-stats.csv")

# ─── 7. RUN PREDICTIONS ──────────────────────────────────────────────────────

predictions = []
for _, row in matches.iterrows():
    pred = predict_match(row["team1"], row["team2"], results)
    pred["match_id"] = row["match_id"]
    pred["date"]     = row["date_"]
    pred["group"]    = row["group_"]
    predictions.append(pred)

df_pred = pd.DataFrame(predictions)

# ─── 8.5 INTEGRATE xG STATS (do this BEFORE odds calibration) ────────────────
from xg_stats import load_match_stats, build_team_xg_profiles, get_prematch_signals, enrich_lambdas_with_xg

stats = load_match_stats(
    "data/international-international-friendlies-matches-2026-to-2026-stats.csv",
    "data/international-world-cup-matches-2026-to-2026-stats.csv",
)

xg_profiles = build_team_xg_profiles(stats, n_recent=5)

# Also need pre-match xG for the upcoming match itself, from the WC file
wc_stats = pd.read_csv("data/international-world-cup-matches-2026-to-2026-stats.csv")
prematch = get_prematch_signals(wc_stats)

df_pred = enrich_lambdas_with_xg(df_pred, xg_profiles, prematch, xg_weight=0.5)

# ─── 8.7 OVERWRITE ELO WITH EXTERNAL 2026 SOURCE ─────────────────────────────
from external_elo import load_external_elo, apply_external_elo, reapply_elo_adjustment_to_lambdas

elo_dict = load_external_elo("data/elo_ratings_wc2026.csv")
df_pred = apply_external_elo(df_pred, elo_dict)
df_pred = reapply_elo_adjustment_to_lambdas(df_pred, elo_weight=ELO_WEIGHT)


# THEN run odds calibration as before — it operates on the now-improved lambdas
# ─── 8. INTEGRATE ODDS ───────────────────────────────────────────────────────

from odds_integration import load_consensus_odds   # add this import

odds = load_consensus_odds(
    "data/cleaned/odds_checker_com.csv",
    "data/cleaned/odds_unibet.csv",
    weights=[0.3, 0.7],
)
df_pred = merge_odds(df_pred, odds)
df_pred = apply_odds_calibration(df_pred, market_weight=0.7)

print_predictions(df_pred)
df_pred = add_results(df_pred, actual, score_col="predicted_score")
df_pred.to_csv("data/out/predictions.csv", index=False)

# ─── 9. MPP SCORE RECOMMENDATIONS ────────────────────────────────────────────
from mpp_bets import compute_mpp_bets, print_mpp_summary

mpp = compute_mpp_bets(df_pred, "data/cleaned/odds_mpp.csv")
print_mpp_summary(mpp)
mpp = add_results(mpp, actual, score_col="recommended_score")
mpp.to_csv("data/out/mpp_bets.csv", index=False)
# Clean version
(
    pl.read_csv("data/out/mpp_bets.csv")
    .select(
        "match_id", "date", "team1", "team2", "recommended_score", "is_contrarian", "risk_level", "expected_value", "naive_score", "naive_ev", "ev_gain_vs_naive", "edge_pct",
        "actual_score","actual_outcome","outcome_correct","score_correct"
    )
    .write_csv("data/out/mpp_bets_clean.csv")
)