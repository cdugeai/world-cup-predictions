-- Load both CSVs as tables first (works in DuckDB, SQLite, or pandas)
-- In DuckDB / pandas SQL:

SELECT DISTINCT team
FROM (
    SELECT team1 AS team FROM 'data/cleaned/matchs_group_clean.csv'
    UNION
    SELECT team2 AS team FROM 'data/cleaned/matchs_group_clean.csv'
) AS all_teams
WHERE
    team NOT IN (
        SELECT DISTINCT home_team FROM 'data/results.csv'
        UNION
        SELECT DISTINCT away_team FROM 'data/results.csv'
    )
ORDER BY team;
