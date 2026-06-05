from bs4 import BeautifulSoup
import polars as pl


def parse_matches(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    matches = []

    for row in soup.find_all('div', class_='RowWrapper_r6ns4d6'):
        # Date/time
        time_tag = row.find('a', class_='StartTimeText_s1kj85jx')
        time = time_tag.get_text(strip=True) if time_tag else None

        # Teams (grab all <p> tags inside BetWrapper)
        bet_wrapper = row.find('a', class_='BetWrapper_btlgb7b')
        teams = [p.get_text(strip=True) for p in bet_wrapper.find_all('p')] if bet_wrapper else []

        # Odds (all buttons inside ButtonsWrapper)
        buttons_wrapper = row.find('div', class_='ButtonsWrapper_b1pxepua')
        odds = [btn.get_text(strip=True) for btn in buttons_wrapper.find_all('button')] if buttons_wrapper else []

        matches.append({
            'time': time,
            'team1': teams[0] if len(teams) > 0 else None,
            'team2': teams[1] if len(teams) > 1 else None,
            'odds': odds
        })

    return matches

# Usage
with open('data/odds_checker_com.html', 'r', encoding='utf-8') as f:
    html = f.read()

odds = []
for match in parse_matches(html):
    odds.append(match)


odds_df = pl.DataFrame(odds)
print(odds_df)
odds_df = (
    odds_df
    .with_columns(
        pl.col("odds").list.to_struct(fields=["odd_1", "odd_null", "odd_2"])
    ).unnest("odds")
    .filter(pl.col('odd_1').is_not_null())
)

print(odds_df)
odds_df.write_csv('data/cleaned/odds_checker_com.csv')