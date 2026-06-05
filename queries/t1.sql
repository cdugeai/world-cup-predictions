CREATE OR REPLACE TABLE matchs_group_clean AS

WITH src AS (
    SELECT
        match AS match_id,
        date AS date_,
        matchup,
        "Group" AS group_
    FROM 'data/matches_group.csv'
),

matchs_group_clean AS (
    SELECT
        match_id,
        date_,
        group_,
        SPLIT_PART(matchup, ' v ', 1) AS team1,
        SPLIT_PART(matchup, ' v ', 2) AS team2
    FROM src
)

SELECT * FROM matchs_group_clean;


COPY (SELECT * FROM matchs_group_clean) TO 'data/cleaned/matchs_group_clean.csv' (HEADER, DELIMITER ',');
