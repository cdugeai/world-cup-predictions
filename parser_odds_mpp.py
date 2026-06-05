from bs4 import BeautifulSoup
import polars as pl


def extract_matches(html):
    soup = BeautifulSoup(html, 'html.parser')
    matches = []

    # Team names use font-family: Futura
    team_divs = soup.find_all('div', style=lambda s: s and 'Futura' in s)
    
    # Percentages contain '%' and use specific style
    pct_divs = soup.find_all('div', style=lambda s: s and 'text-align: center' in (s or ''))
    percentages = [d.get_text(strip=True) for d in pct_divs if d.get_text(strip=True).endswith('%')]

    # Pair teams and odds (2 teams + 3 odds per match)
    for i in range(0, len(team_divs) - 1, 2):
        j = (i // 2) * 3  # 3 odds per match
        if j + 2 < len(percentages):
            matches.append({
                'team1':      team_divs[i].get_text(strip=True),
                'team2':      team_divs[i + 1].get_text(strip=True),
                'team1_win':  percentages[j],
                'draw':       percentages[j + 1],
                'team2_win':  percentages[j + 2],
            })

    return matches

# Example
with open('data/mpp_odds.html', 'r') as f:
    html = f.read()

odds = []
for m in extract_matches(html):
    odds.append(m)


odds_df = pl.DataFrame(odds).select(
    "team1", "team2",
    (pl.col('team1_win').str.replace('%', '').cast(pl.Int8) / 100).alias("team1_win_pct"),
    (pl.col('draw').str.replace('%', '').cast(pl.Int8) / 100).alias("draw_pct"),
    (pl.col('team2_win').str.replace('%', '').cast(pl.Int8) / 100).alias("team2_win_pct"),
)

odds_df.write_csv('data/cleaned/odds_mpp.csv')

print(odds_df)