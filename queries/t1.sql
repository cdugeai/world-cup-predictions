CREATE OR REPLACE TABLE matchs_group_clean AS

WITH src AS (
    SELECT
        match AS match_id,
        date AS date_,
        matchup,
        "Group" AS group_
    FROM 'data/matches_group.csv'
),

aliases AS (
    SELECT
        alias,
        canonical
    FROM 'data/cleaned/team_aliases.csv'
),

matchs_group AS (
    SELECT
        match_id,
        date_,
        group_,
        SPLIT_PART(matchup, ' v ', 1) AS team1,
        SPLIT_PART(matchup, ' v ', 2) AS team2
    FROM src
),

matchs_group_aliased AS (
    SELECT
        match_id,
        date_,
        group_,
        team1,
        team2,
        a1.canonical AS team1_canonical,
        a2.canonical AS team2_canonical
    FROM matchs_group AS mg
    LEFT JOIN aliases AS a1 ON mg.team1 = a1.alias
    LEFT JOIN aliases AS a2 ON mg.team2 = a2.alias
),

matchs_group_clean AS (
    SELECT
        match_id,
        date_,
        group_,
        COALESCE(team1_canonical, team1) AS team1,
        COALESCE(team2_canonical, team2) AS team2
    FROM matchs_group_aliased
)


SELECT * FROM matchs_group_clean
ORDER BY match_id;


COPY (SELECT * FROM matchs_group_clean) TO 'data/cleaned/matchs_group_clean.csv' (HEADER, DELIMITER ',');
