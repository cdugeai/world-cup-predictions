here is my project directory:

.
├── data
│ ├── cleaned
│ │ └── matchs_group_clean.csv
│ ├── fifa_mens_rank.csv
│ ├── results.csv
│ ├── shootouts.csv
│ └── sources.md
├── main.py
├── prompt.md
├── pyproject.toml
├── README.md
└── uv.lock

Here is an extract of the schema of each file:

fifa_mens_rank.csv:

date,semester,rank,team,acronym,total.points,previous.points,diff.points
2024,2,1,Argentina,ARG,1867.25,1883.5,-16.25
2024,2,2,France,FRA,1859.78,1859.85,-0.0699999999999363
2024,2,3,Spain,ESP,1853.27,1844.33,8.94000000000005

results.csv:

date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE
1873-03-08,England,Scotland,4,2,Friendly,London,England,FALSE
1874-03-07,Scotland,England,2,1,Friendly,Glasgow,Scotland,FALSE
1875-03-06,England,Scotland,2,2,Friendly,London,England,FALSE

shootouts.csv:

date,home_team,away_team,winner,first_shooter
1967-08-22,India,Taiwan,Taiwan,
1971-11-14,South Korea,Vietnam Republic,South Korea,
1972-05-07,South Korea,Iraq,Iraq,

List of matches to predict (warn, names can be misaligned with previous datasets):

data/cleaned/matchs_group_clean.csv

```
match_id,date_,group_,team1,team2
1,11-Jun-26,A,Mexico,South Africa
2,11-Jun-26,A,South Korea,Czechia
3,12-Jun-26,B,Canada,Bosnia and Herzegovina
4,12-Jun-26,D,USA,Paraguay
```

What code
