from bs4 import BeautifulSoup
import polars as pl


def extract_matches(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Team names
    team_divs = soup.find_all('div', style=lambda s: s and 'Futura' in s)

    # Points: inner div of the circle (r-131xog0 > div)
    pts_divs = soup.select('div.r-131xog0 > div')
    # Percentages: unique class r-1bymd8e
    pct_divs = soup.select('div.r-1bymd8e')

    points = [d.get_text(strip=True) for d in pts_divs]
    pcts   = [d.get_text(strip=True) for d in pct_divs]

    print(f"Teams: {len(team_divs)}, Points: {len(points)}, Pcts: {len(pcts)}")  # sanity check

    matches = []
    for i in range(0, len(team_divs) - 1, 2):
        j = (i // 2) * 3
        if j + 2 < len(points):
            matches.append({
                'team1':         team_divs[i].get_text(strip=True),
                'team2':         team_divs[i + 1].get_text(strip=True),
                'team1_win_pct': pcts[j],
                'team1_points':  points[j],
                'draw_pct':      pcts[j + 1],
                'draw_points':   points[j + 1],
                'team2_win_pct': pcts[j + 2],
                'team2_points':  points[j + 2],
            })

    return matches


with open('data/mpp_odds.html', 'r') as f:
    html = f.read()

odds = extract_matches(html)

odds_df = pl.DataFrame(odds).select(
    "team1", "team2",
    (pl.col('team1_win_pct').str.replace('%', '').cast(pl.Int8) / 100).alias("team1_win_pct"),
    pl.col('team1_points').cast(pl.Int16),
    (pl.col('draw_pct').str.replace('%', '').cast(pl.Int8) / 100).alias("draw_pct"),
    pl.col('draw_points').cast(pl.Int16),
    (pl.col('team2_win_pct').str.replace('%', '').cast(pl.Int8) / 100).alias("team2_win_pct"),
    pl.col('team2_points').cast(pl.Int16),
)

odds_df.write_csv('data/cleaned/odds_mpp.csv')
print(odds_df)