i play pronostic game for fifa world cup. The points for correct result is here team1_points, team2_points, draw_points.

odds_mpp.csv

team1,team2,team1_win_pct,team1_points,draw_pct,draw_points,team2_win_pct,team2_points
Mexico,South Africa,0.8200000000000001,49,0.1,125,0.08,148
South Korea,Czech Republic,0.31,96,0.5,107,0.19,91

I have a few csv with info on past matchs

international-international-friendlies-matches-2026-to-2026-stats.csv

timestamp,date_GMT,status,attendance,home_team_name,away_team_name,referee,Game Week,Pre-Match PPG (Home),Pre-Match PPG (Away),home_ppg,away_ppg,home_team_goal_count,away_team_goal_count,total_goal_count,total_goals_at_half_time,home_team_goal_count_half_time,away_team_goal_count_half_time,home_team_goal_timings,away_team_goal_timings,home_team_corner_count,away_team_corner_count,home_team_yellow_cards,home_team_red_cards,away_team_yellow_cards,away_team_red_cards,home_team_first_half_cards,home_team_second_half_cards,away_team_first_half_cards,away_team_second_half_cards,home_team_shots,away_team_shots,home_team_shots_on_target,away_team_shots_on_target,home_team_shots_off_target,away_team_shots_off_target,home_team_fouls,away_team_fouls,home_team_possession,away_team_possession,Home Team Pre-Match xG,Away Team Pre-Match xG,team_a_xg,team_b_xg,average_goals_per_match_pre_match,btts_percentage_pre_match,over_15_percentage_pre_match,over_25_percentage_pre_match,over_35_percentage_pre_match,over_45_percentage_pre_match,over_15_HT_FHG_percentage_pre_match,over_05_HT_FHG_percentage_pre_match,over_15_2HG_percentage_pre_match,over_05_2HG_percentage_pre_match,average_corners_per_match_pre_match,average_cards_per_match_pre_match,odds_ft_home_team_win,odds_ft_draw,odds_ft_away_team_win,odds_ft_over15,odds_ft_over25,odds_ft_over35,odds_ft_over45,odds_btts_yes,odds_btts_no,stadium_name
1768705200,Jan 18 2026 - 3:00am,complete,N/A,Canada,Guatemala,Rosendo Mendoza,N/A,0.00,0.00,1.80,0.00,1,0,1,0,0,0,"66","",12,5,2,0,1,0,1,1,1,0,14,6,7,3,7,3,12,6,58,42,0.00,0.00,0.00,0.00,0.00,0,0,0,0,0,0,0,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,Banc of California Stadium

and matches of current competition:

international-world-cup-matches-2026-to-2026-stats.csv
timestamp,date_GMT,status,attendance,home_team_name,away_team_name,referee,Game Week,Pre-Match PPG (Home),Pre-Match PPG (Away),home_ppg,away_ppg,home_team_goal_count,away_team_goal_count,total_goal_count,total_goals_at_half_time,home_team_goal_count_half_time,away_team_goal_count_half_time,home_team_goal_timings,away_team_goal_timings,home_team_corner_count,away_team_corner_count,home_team_yellow_cards,home_team_red_cards,away_team_yellow_cards,away_team_red_cards,home_team_first_half_cards,home_team_second_half_cards,away_team_first_half_cards,away_team_second_half_cards,home_team_shots,away_team_shots,home_team_shots_on_target,away_team_shots_on_target,home_team_shots_off_target,away_team_shots_off_target,home_team_fouls,away_team_fouls,home_team_possession,away_team_possession,Home Team Pre-Match xG,Away Team Pre-Match xG,team_a_xg,team_b_xg,average_goals_per_match_pre_match,btts_percentage_pre_match,over_15_percentage_pre_match,over_25_percentage_pre_match,over_35_percentage_pre_match,over_45_percentage_pre_match,over_15_HT_FHG_percentage_pre_match,over_05_HT_FHG_percentage_pre_match,over_15_2HG_percentage_pre_match,over_05_2HG_percentage_pre_match,average_corners_per_match_pre_match,average_cards_per_match_pre_match,odds_ft_home_team_win,odds_ft_draw,odds_ft_away_team_win,odds_ft_over15,odds_ft_over25,odds_ft_over35,odds_ft_over45,odds_btts_yes,odds_btts_no,stadium_name
1781204400,Jun 11 2026 - 7:00pm,complete,80824,Mexico,South Africa,N/A,1,0.00,0.00,3.00,0.00,2,0,2,1,1,0,"9,67","",3,1,1,1,2,2,1,1,1,3,16,3,4,2,12,1,12,11,61,39,0.00,0.00,1.50,0.54,0.00,0,0,0,0,0,0,0,0,0,0.00,0.00,1.40,4.40,7.75,1.35,2.20,4.00,8.00,2.38,1.50,Estadio Azteca

and fifa elo rating

elo_ratings_wc2026.csv

year,snapshot_date,country,rank,country_code,rating,rank_max,rating_max,rank_avg,rating_avg,rank_min,rating_min,matches_total,matches_home,matches_away,matches_neutral,wins,losses,draws,goals_for,goals_against,confederation,is_host
2026,2026-12-31,Spain,1,ES,2165,1,2189,7,1946,19,1805,780,340,302,138,461,138,181,1591,697,UEFA,0
2026,2026-12-31,Argentina,2,AR,2113,1,2172,5,1987,26,1751,1109,381,419,309,610,228,271,2112,1136,CONMEBOL,0

odds from bookmaker:

odds_unibet.csv

time,team1,team2,odd_1,odd_null,odd_2
21:00,Mexico,South Africa,21/50,10/3,15/2
04:00,South Korea,Czechia,8/5,2/1,15/8
21:00,Canada,Bosnia & Herzegovina,4/5,5/2,7/2

other odds sources (less reliable):

odds_checker_com.csv

time,team1,team2,odd_1,odd_null,odd_2
20:00,Mexico,South Africa,4/9,7/2,8/1
03:00,South Korea,Czech Republic,33/20,11/5,9/5

IDs of matches:

matches_group.csv

"Match","Date","Time (EST)","Time (Local)","Matchup","Group","Venue","City"
"1","11-Jun-26","15:00","13:00","Mexico v South Africa","A","Estadio Azteca","Mexico City"
"2","11-Jun-26","22:00","20:00","South Korea v Czechia","A","Estadio Akron","Guadalajara"

and mapping to make sure we have consistent names:

team_aliases.csv

alias,canonical
United States,USA
Türkiye,Turkey

Build python algorithm to gives recommendation in this final format:

mpp_bets_v2.py

match_id,date,team1,team2,recommended_score,recommended_outcome,p_outcome,p_exact_score,points_if_correct,expected_value,platform_favourite,platform_p,naive_score,naive_ev,ev_gain_vs_naive,edge_pct,is_contrarian,risk_level,ev_team1_win,ev_draw,ev_team2_win,score_team1_win,score_draw,score_team2_win,actual_score,actual_outcome,outcome_correct,score_correct
1,11-Jun-26,Mexico,South Africa,1-0,team1_win,0.473,0.192,49,23.18,team1_win,0.82,1-0,40.18,-17.0,21.8,False,FOLLOW_FAVOURITE,23.18,39.71,30.95,1-0,0-0,0-1,2-0,team1_win,True,False
2,11-Jun-26,South Korea,Czech Republic,1-1,draw,0.243,0.114,107,26.0,draw,0.5,1-1,53.5,-27.5,15.7,False,FOLLOW_FAVOURITE,44.83,26.0,26.22,2-1,1-1,1-2,2-1,team1_win,False,False
3,12-Jun-26,Canada,Bosnia,0-1,team2_win,0.49,0.142,125,61.26,team1_win,0.59,1-0,38.35,22.91,31.0,True,VALUE_CONTRARIAN,15.42,31.83,61.26,1-0,1-1,0-1,1-1,draw,False,False
