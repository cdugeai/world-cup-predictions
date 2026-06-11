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

def _detect_format(odds: pd.DataFrame) -> str:
    """
    Auto-detect odds format from column names:
      - 'percentage' : team1_win_pct / draw_pct / team2_win_pct
      - 'fractional' : odd_1 / odd_null / odd_2
    """
    cols = set(odds.columns)
    if {"team1_win_pct", "draw_pct", "team2_win_pct"}.issubset(cols):
        return "percentage"
    if {"odd_1", "odd_null", "odd_2"}.issubset(cols):
        return "fractional"
    raise ValueError(
        "Cannot detect odds format. Expected columns:\n"
        "  percentage format: team1_win_pct, draw_pct, team2_win_pct\n"
        "  fractional format: odd_1, odd_null, odd_2"
    )

def load_odds(filepath: str) -> pd.DataFrame:
    """
    Load odds CSV — auto-detects format (percentage or fractional).

    Percentage format:  team1_win_pct / draw_pct / team2_win_pct  (values 0-1 or 0-100)
    Fractional format:  odd_1 / odd_null / odd_2                  (values like '33/20')

    Always outputs: mkt_win1, mkt_draw, mkt_win2 (normalized, sum to 1.0) + margin.
    """
    odds = pd.read_csv(filepath)
    fmt = _detect_format(odds)

    if fmt == "percentage":
        p1 = pd.to_numeric(odds["team1_win_pct"], errors="coerce")
        pd_ = pd.to_numeric(odds["draw_pct"],      errors="coerce")
        p2 = pd.to_numeric(odds["team2_win_pct"],  errors="coerce")

        # Support both 0–1 and 0–100 ranges
        if p1.max() > 1.5:
            p1, pd_, p2 = p1 / 100, pd_ / 100, p2 / 100

        odds["p1_raw"], odds["pd_raw"], odds["p2_raw"] = p1, pd_, p2

        # Alias display columns to match fractional convention
        odds["odd_1"]    = odds["team1_win_pct"]
        odds["odd_null"] = odds["draw_pct"]
        odds["odd_2"]    = odds["team2_win_pct"]

    else:  # fractional
        odds["p1_raw"] = odds["odd_1"].apply(fractional_to_prob)
        odds["pd_raw"] = odds["odd_null"].apply(fractional_to_prob)
        odds["p2_raw"] = odds["odd_2"].apply(fractional_to_prob)

    odds["margin"] = odds["p1_raw"] + odds["pd_raw"] + odds["p2_raw"] - 1.0

    normalized = odds.apply(
        lambda r: normalize_probs(r["p1_raw"], r["pd_raw"], r["p2_raw"]), axis=1
    )
    odds[["mkt_win1", "mkt_draw", "mkt_win2"]] = pd.DataFrame(
        normalized.tolist(), index=odds.index
    )

    print(f"  Loaded {len(odds)} odds rows  [format: {fmt}]")
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


# ─── TEAM NAME ALIASES ───────────────────────────────────────────────────────

# Default path — override by passing aliases_path= to merge_odds()
DEFAULT_ALIASES_PATH = "data/cleaned/team_aliases.csv"

def load_aliases(filepath: str = DEFAULT_ALIASES_PATH) -> dict[str, str]:
    """
    Load manual alias -> canonical name mappings from CSV.
    Expected columns: alias, canonical

    Example data/team_aliases.csv:
        alias,canonical
        USA,United States
        Turkey,Türkiye
        Czechia,Czech Republic
        Bosnia,Bosnia and Herzegovina
        Korea Republic,South Korea
        IR Iran,Iran

    Returns a dict {alias_lower: canonical} for case-insensitive lookup.
    Missing file is silently ignored (returns empty dict).
    """
    try:
        df = pd.read_csv(filepath)
        aliases = {
            str(row["alias"]).strip().lower(): str(row["canonical"]).strip()
            for _, row in df.iterrows()
        }
        print(f"  Loaded {len(aliases)} team aliases from {filepath}")
        return aliases
    except FileNotFoundError:
        return {}


# ─── MERGE ODDS INTO PREDICTIONS ─────────────────────────────────────────────

def _fuzzy_match(name: str, candidates: list[str], threshold: float = 0.75) -> str | None:
    """
    Match a team name to the closest candidate using character n-gram similarity.
    Returns the best match if above threshold, else None.
    """
    from difflib import SequenceMatcher
    name_l = name.lower().strip()
    best_score, best_match = 0.0, None
    for c in candidates:
        score = SequenceMatcher(None, name_l, c.lower().strip()).ratio()
        # Bonus: substring match (e.g. 'Bosnia' inside 'Bosnia and Herzegovina')
        if name_l in c.lower() or c.lower() in name_l:
            score = max(score, 0.85)
        if score > best_score:
            best_score, best_match = score, c
    return best_match if best_score >= threshold else None

def merge_odds(
    df_pred: pd.DataFrame,
    df_odds: pd.DataFrame,
    aliases_path: str = DEFAULT_ALIASES_PATH,
) -> pd.DataFrame:
    """
    Join odds to predictions on team names using:
      1. Manual alias table (data/team_aliases.csv) — checked first, always wins
      2. Fuzzy matching — fallback for names not in the alias table

    To fix a ⚠ warning, just add a row to data/team_aliases.csv — no code changes needed.
    Unmatched rows keep model-only probabilities.
    """
    df_pred = df_pred.copy()
    df_odds  = df_odds.copy()

    aliases = load_aliases(aliases_path)  # {alias_lower: canonical}
    pred_teams = list(df_pred["team1"].unique()) + list(df_pred["team2"].unique())

    # Build a name mapping: odds name -> prediction name
    # Priority: alias table > fuzzy match
    name_fixes: dict[str, str] = {}
    for col in ["team1", "team2"]:
        for odds_name in df_odds[col].unique():
            if odds_name in pred_teams:
                continue  # exact match, nothing to do

            alias_hit = aliases.get(odds_name.strip().lower())
            if alias_hit:
                name_fixes[odds_name] = alias_hit
                print(f"  ✓ Alias match:  '{odds_name}' -> '{alias_hit}'")
            else:
                match = _fuzzy_match(odds_name, pred_teams)
                if match:
                    name_fixes[odds_name] = match
                    print(f"  ↳ Fuzzy match: '{odds_name}' -> '{match}'")
                else:
                    print(f"  ⚠ No match for '{odds_name}' — add to data/team_aliases.csv to fix")

    # Apply fixes to odds team names
    for col in ["team1", "team2"]:
        df_odds[col] = df_odds[col].replace(name_fixes)

    odds_cols = ["team1", "team2", "odd_1", "odd_null", "odd_2",
                 "mkt_win1", "mkt_draw", "mkt_win2", "margin"]

    merged = df_pred.merge(df_odds[odds_cols], on=["team1", "team2"], how="left")

    matched = merged["mkt_win1"].notna().sum()
    print(f"  Odds matched: {matched}/{len(merged)} matches")

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
                  f"  (vig: {r['margin']:+.1%})")
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

def load_consensus_odds(
    *filepaths: str,
    weights: list[float] | None = None,
    aliases_path: str = DEFAULT_ALIASES_PATH,
) -> pd.DataFrame:
    """
    Load multiple odds CSVs with the same format and merge into consensus probabilities.

    Parameters
    ----------
    *filepaths   : any number of CSV paths, e.g.
                   load_consensus_odds("data/odds_checker.csv", "data/betexplorer.csv")
    weights      : optional per-source weights, e.g. [0.6, 0.4]
                   defaults to equal weighting across all sources
    aliases_path : path to team_aliases.csv — same file used by merge_odds()

    Returns the same structure as load_odds() — drop-in replacement.
    Row count always equals the number of matches in the FIRST source file.

    Name resolution priority (same as merge_odds):
      1. Exact match         → used as-is
      2. data/team_aliases.csv → ✓ Alias match
      3. Fuzzy match         → ↳ Fuzzy match
      4. No match            → ⚠ warning, row gets NaN for that source
    """
    sources = [load_odds(fp) for fp in filepaths]

    if weights is None:
        weights = [1.0 / len(sources)] * len(sources)
    else:
        if len(weights) != len(sources):
            raise ValueError(f"Got {len(filepaths)} files but {len(weights)} weights")
        total = sum(weights)
        weights = [w / total for w in weights]

    for i, (src, fp) in enumerate(zip(sources, filepaths)):
        src_name = fp.split("/")[-1].replace(".csv", "")
        print(f"  Source {i+1}: {src_name} — {len(src)} matches  (weight {weights[i]:.0%})")

    prob_cols = ["mkt_win1", "mkt_draw", "mkt_win2"]

    # Anchor team names come from source 1 — all other sources are normalized to match
    anchor_names = list(set(sources[0]["team1"]) | set(sources[0]["team2"]))
    aliases = load_aliases(aliases_path)  # {alias_lower: canonical}

    def normalize_to_anchor(src: pd.DataFrame, src_label: str) -> pd.DataFrame:
        """
        Remap team names in src to match source-1 names.
        Identical resolution logic as merge_odds():
          1. Exact match in anchor names  → keep as-is
          2. Alias table hit              → ✓ Alias match
          3. Fuzzy match above threshold  → ↳ Fuzzy match
          4. No match                     → ⚠ warning
        """
        src = src.copy()
        name_fixes: dict[str, str] = {}

        for col in ["team1", "team2"]:
            for name in src[col].unique():
                if name in anchor_names or name in name_fixes:
                    continue  # already resolved

                alias_hit = aliases.get(name.strip().lower())
                if alias_hit and alias_hit in anchor_names:
                    name_fixes[name] = alias_hit
                    print(f"  [{src_label}] ✓ Alias match:  '{name}' -> '{alias_hit}'")
                else:
                    fuzzy_hit = _fuzzy_match(name, anchor_names)
                    if fuzzy_hit:
                        name_fixes[name] = fuzzy_hit
                        print(f"  [{src_label}] ↳ Fuzzy match: '{name}' -> '{fuzzy_hit}'")
                    else:
                        print(f"  [{src_label}] ⚠ No match for '{name}' — add to team_aliases.csv to fix")

        for col in ["team1", "team2"]:
            src[col] = src[col].replace(name_fixes)

        return src

    # Merge all sources into source 1, left join to preserve row count
    merged = sources[0].copy()

    for i, src in enumerate(sources[1:], start=1):
        src_label = filepaths[i].split("/")[-1].replace(".csv", "")
        src_aligned = normalize_to_anchor(src, src_label)

        merged = merged.merge(
            src_aligned[["team1", "team2"] + prob_cols + ["margin"]],
            on=["team1", "team2"],
            how="left",                        # anchor to source 1 — no extra rows
            suffixes=("", f"_s{i}"),
        )

    # Weighted average per probability column, skipping NaN sources per row
    w_array = np.array(weights)

    for col in prob_cols:
        source_cols = [col] + [f"{col}_s{i}" for i in range(1, len(sources))]
        col_data = merged[source_cols]

        def weighted_row(row, w=w_array):
            vals = row.values.astype(float)
            mask = ~np.isnan(vals)
            if not mask.any():
                return np.nan
            w_valid = w[mask] / w[mask].sum()  # renormalize for partial coverage
            return float(np.dot(vals[mask], w_valid))

        merged[col] = col_data.apply(weighted_row, axis=1)

    # Re-normalize consensus probs to sum to exactly 1.0
    row_totals = merged[prob_cols].sum(axis=1)
    for col in prob_cols:
        merged[col] = merged[col] / row_totals

    # Average margin across sources
    margin_cols = [c for c in merged.columns if c == "margin" or c.startswith("margin_s")]
    merged["margin"] = merged[margin_cols].mean(axis=1)

    # Coverage report
    s1_only_cols = [f"mkt_win1_s{i}" for i in range(1, len(sources)) if f"mkt_win1_s{i}" in merged.columns]
    n_full = int(merged[s1_only_cols].notna().all(axis=1).sum()) if s1_only_cols else len(merged)
    print(f"  ────────────────────────────────────")
    print(f"  Consensus rows  : {len(merged)}  ✓ (matches source-1 count)")
    print(f"  All sources     : {n_full}/{len(merged)}")
    print(f"  Single-source   : {len(merged) - n_full}  (source-1 probs used as fallback)")

    # Drop per-source suffix columns
    drop_cols = [c for c in merged.columns
                 if any(c.endswith(f"_s{i}") for i in range(1, len(sources)))]
    return merged.drop(columns=drop_cols)