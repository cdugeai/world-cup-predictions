SELECT 
    a.team1, a.team2, 
    a.predicted_score, 
    a.score_cal, 
    a.model_vs_market, 
    b.bet_recommendation,
    b.bet_outcome,
    b.predicted_score,
    b.edge_pct
    --a.* 
from "data/out/predictions.csv" as a
left join "data/out/betting_recommendations.csv" as b on a.match_id=b.match_id
