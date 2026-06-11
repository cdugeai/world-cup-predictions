import csv
from bs4 import BeautifulSoup

def parse_odds(html_content: str) -> list[dict]:
    soup = BeautifulSoup(html_content, "html.parser")
    results = []

    cards = soup.find_all("div", class_="emaeyjf0 css-17bdkdg e1pg1gpv0")

    for card in cards:
        # Time
        time_el = card.find(attrs={"testid": "contestTimeInCard"})
        time = time_el.get_text(strip=True) if time_el else ""

        # Teams
        teams = card.find_all(attrs={"testid": "teamNamesInHeader"})
        team1 = teams[0].get_text(strip=True) if len(teams) > 0 else ""
        team2 = teams[1].get_text(strip=True) if len(teams) > 1 else ""

        # Odds (in order: win1, draw, win2)
        odds_els = card.find_all(attrs={"testid": "propositionOptionValue"})
        odd_1    = odds_els[0].get_text(strip=True) if len(odds_els) > 0 else ""
        odd_null = odds_els[1].get_text(strip=True) if len(odds_els) > 1 else ""
        odd_2    = odds_els[2].get_text(strip=True) if len(odds_els) > 2 else ""

        results.append({
            "time": time,
            "team1": team1,
            "team2": team2,
            "odd_1": odd_1,
            "odd_null": odd_null,
            "odd_2": odd_2,
        })

    return results


def save_to_csv(rows: list[dict], output_path: str) -> None:
    fieldnames = ["time", "team1", "team2", "odd_1", "odd_null", "odd_2"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    with open("data/odds_betclic.html", "r", encoding="utf-8") as f:
        html = f.read()

    rows = parse_odds(html)
    save_to_csv(rows, "data/cleaned/odds_betclic.csv")
    print(f"Parsed {len(rows)} matches to csv")