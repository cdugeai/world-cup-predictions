import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BettingOptimizer:
    """
    Kelly Criterion based betting recommendation system for probabilistic sports predictions.
    """
    
    def __init__(self, kelly_fraction: float = 0.25, min_edge_threshold: float = 0.05):
        """
        Initialize the optimizer.
        
        Args:
            kelly_fraction: Fractional Kelly to use (0.25 = quarter Kelly, safer)
            min_edge_threshold: Minimum edge required to recommend a bet (5% = 0.05)
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge_threshold = min_edge_threshold
    
    def calculate_implied_probability(self, odds: float) -> float:
        """
        Convert decimal odds to implied probability.
        
        Args:
            odds: Decimal odds (e.g., 2.5)
        
        Returns:
            Implied probability (0-1)
        """
        return 1 / odds
    
    def odds_to_decimal(self, odds_str: str) -> Optional[float]:
        """
        Convert fractional odds (e.g., "4/9") to decimal odds.
        
        Args:
            odds_str: Fractional odds as string (e.g., "4/9")
        
        Returns:
            Decimal odds or None if invalid
        """
        try:
            if pd.isna(odds_str) or odds_str == '':
                return None
            parts = str(odds_str).split('/')
            numerator = float(parts[0])
            denominator = float(parts[1])
            return (numerator + denominator) / denominator
        except (ValueError, IndexError):
            return None
    
    def kelly_criterion(self, 
                       predicted_prob: float, 
                       decimal_odds: float, 
                       max_bet_pct: float = 0.05) -> Dict:
        """
        Calculate Kelly Criterion bet recommendation.
        
        Formula: f* = (bp - q) / b
        where:
        - b = odds - 1 (net odds)
        - p = predicted probability
        - q = 1 - p
        
        Args:
            predicted_prob: Your model's predicted probability of outcome
            decimal_odds: Decimal odds offered (e.g., 2.5)
            max_bet_pct: Maximum percentage of bankroll to risk (safety cap)
        
        Returns:
            Dictionary with Kelly calculation details
        """
        if predicted_prob <= 0 or predicted_prob >= 1 or decimal_odds <= 1:
            return {
                'kelly_fraction': 0,
                'kelly_pct': 0,
                'fractional_kelly': 0,
                'fractional_kelly_pct': 0,
                'edge': 0,
                'recommendation': 'SKIP'
            }
        
        # Net odds
        b = decimal_odds - 1
        p = predicted_prob
        q = 1 - p
        
        # Full Kelly
        kelly_full = (b * p - q) / b
        kelly_full = max(0, kelly_full)  # No negative bets
        
        # Fractional Kelly (safer)
        kelly_frac = kelly_full * self.kelly_fraction
        kelly_frac = min(kelly_frac, max_bet_pct)  # Cap at max
        
        # Calculate edge
        market_prob = self.calculate_implied_probability(decimal_odds)
        edge = p - market_prob
        
        # Recommendation logic
        if edge <= self.min_edge_threshold:
            recommendation = 'SKIP - Insufficient edge'
        elif kelly_frac <= 0.001:
            recommendation = 'SKIP - Kelly too small'
        else:
            recommendation = 'BET'
        
        return {
            'kelly_fraction': kelly_full,
            'kelly_pct': kelly_full * 100,
            'fractional_kelly': kelly_frac,
            'fractional_kelly_pct': kelly_frac * 100,
            'edge': edge,
            'edge_pct': edge * 100,
            'market_prob': market_prob,
            'recommendation': recommendation
        }
    
    def calculate_expected_value(self, 
                                predicted_prob: float, 
                                decimal_odds: float, 
                                bet_amount: float = 1.0) -> float:
        """
        Calculate expected value of a bet.
        
        EV = (P_win × Payout) - (P_loss × Stake)
        
        Args:
            predicted_prob: Your model's predicted probability
            decimal_odds: Decimal odds
            bet_amount: Amount to bet (default 1.0)
        
        Returns:
            Expected value
        """
        payout = (decimal_odds - 1) * bet_amount
        loss = bet_amount
        ev = (predicted_prob * payout) - ((1 - predicted_prob) * loss)
        return ev
    
    def process_predictions(self, 
                           predictions_df: pd.DataFrame, 
                           points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process predictions and calculate betting recommendations.
        
        Args:
            predictions_df: DataFrame with predictions and odds
            points_df: DataFrame with point values for each outcome
        
        Returns:
            DataFrame with betting recommendations
        """
        results = []
        
        for idx, row in predictions_df.iterrows():
            match_id = row.get('match_id', idx)
            team1 = row['team1']
            team2 = row['team2']
            date = row.get('date', '')
            
            # Get point values for this match
            points_row = points_df[
                ((points_df['team1'] == team1) & (points_df['team2'] == team2)) |
                ((points_df['team1'].str.contains(team1, na=False)) & 
                 (points_df['team2'].str.contains(team2, na=False)))
            ]
            
            if points_row.empty:
                logger.warning(f"No point data found for {team1} vs {team2}")
                continue
            
            points_row = points_row.iloc[0]
            
            # Extract probabilities and odds
            p_win1 = row['p_win1_final']
            p_draw = row['p_draw_final']
            p_win2 = row['p_win2_final']
            
            odd_1_str = row.get('odd_1', '')
            odd_null_str = row.get('odd_null', '')
            odd_2_str = row.get('odd_2', '')
            
            predicted_score = row.get('predicted_score', '')
            
            # Convert fractional odds to decimal
            odd_1 = self.odds_to_decimal(odd_1_str)
            odd_null = self.odds_to_decimal(odd_null_str)
            odd_2 = self.odds_to_decimal(odd_2_str)
            
            # Calculate Kelly recommendations for each outcome
            rec_win1 = self.kelly_criterion(p_win1, odd_1) if odd_1 else None
            rec_draw = self.kelly_criterion(p_draw, odd_null) if odd_null else None
            rec_win2 = self.kelly_criterion(p_win2, odd_2) if odd_2 else None
            
            # Find best recommendation
            recommendations = [
                ('win1', rec_win1, points_row['team1_points'], p_win1, odd_1) if rec_win1 else None,
                ('draw', rec_draw, points_row['draw_points'], p_draw, odd_null) if rec_draw else None,
                ('win2', rec_win2, points_row['team2_points'], p_win2, odd_2) if rec_win2 else None,
            ]
            recommendations = [r for r in recommendations if r is not None]
            
            # Sort by edge, filtered for actual bets
            bettable = [r for r in recommendations if r[1]['recommendation'] == 'BET']
            best_rec = max(bettable, key=lambda x: x[1]['edge']) if bettable else None
            
            # Prepare result row
            result = {
                'match_id': match_id,
                'date': date,
                'team1': team1,
                'team2': team2,
                'predicted_score': predicted_score,
                'p_win1': round(p_win1, 3),
                'p_draw': round(p_draw, 3),
                'p_win2': round(p_win2, 3),
                'odd_1': odd_1,
                'odd_draw': odd_null,
                'odd_2': odd_2,
                'bet_recommendation': 'SKIP',
                'bet_outcome': '',
                'bet_odds': '',
                'bet_probability': '',
                'bet_points_value': '',
                'kelly_pct': '',
                'edge_pct': '',
                'expected_value': '',
                'risk_level': '',
            }
            
            if best_rec:
                outcome, rec, points, prob, odds = best_rec
                ev = self.calculate_expected_value(prob, odds)
                
                # Determine risk level based on odds and edge
                risk_level = self._assess_risk_level(odds, rec['edge'], prob)
                
                result.update({
                    'bet_recommendation': 'BET',
                    'bet_outcome': f'{team1} win' if outcome == 'win1' else ('Draw' if outcome == 'draw' else f'{team2} win'),
                    'bet_odds': round(odds, 2),
                    'bet_probability': round(prob, 3),
                    'bet_points_value': int(points),
                    'kelly_pct': round(rec['fractional_kelly_pct'], 2),
                    'edge_pct': round(rec['edge_pct'], 2),
                    'expected_value': round(ev, 4),
                    'risk_level': risk_level,
                })
            else:
                # Provide details on why skipped
                if recommendations:
                    best_skip = max(recommendations, key=lambda x: x[1]['edge'])
                    _, rec, _, prob, odds = best_skip
                    result.update({
                        'edge_pct': round(rec['edge_pct'], 2),
                        'risk_level': 'HIGH_EDGE_NEEDED'
                    })
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _assess_risk_level(self, odds: float, edge: float, probability: float) -> str:
        """
        Assess relative risk level of a bet.
        
        Args:
            odds: Decimal odds
            edge: Your edge percentage
            probability: Your predicted probability
        
        Returns:
            Risk level string
        """
        if probability >= 0.85:
            return 'LOW'
        elif probability >= 0.70 and edge >= 0.10:
            return 'MEDIUM'
        elif probability >= 0.60 and edge >= 0.15:
            return 'MEDIUM-HIGH'
        elif edge >= 0.20:
            return 'HIGH'
        else:
            return 'VERY_HIGH'


def main():
    """Example usage."""
    
    # Load CSVs
    predictions = pd.read_csv('data/out/predictions.csv')
    points = pd.read_csv('data/cleaned/odds_mpp.csv')
    
    # Initialize optimizer
    optimizer = BettingOptimizer(
        kelly_fraction=0.25,  # Quarter Kelly for safety
        min_edge_threshold=0.05  # 5% minimum edge
    )
    
    # Generate recommendations
    recommendations = optimizer.process_predictions(predictions, points)
    
    # Save output
    OUTFILE = 'data/out/betting_recommendations.csv'
    recommendations.to_csv(OUTFILE, index=False)
    logger.info(f"Saved recommendations to betting_recommendations.csv")
    
    # Clean version
    recommendations_pl = pl.read_csv(OUTFILE)
    predictions_pl = pl.DataFrame(predictions.to_dict('records'))

    recommendations_clean = (
        predictions_pl.select("match_id", "team1", "team2", "predicted_score", "score_cal", "model_vs_market")
        .join(recommendations_pl.select("match_id", "bet_recommendation", "bet_outcome", "predicted_score", "edge_pct"), how="left", on="match_id")
    )

    recommendations_clean.write_csv('data/out/betting_recommendations_clean.csv')
    


    # Print summary
    print("\n" + "="*80)
    print("BETTING RECOMMENDATIONS SUMMARY")
    print("="*80)
    print(recommendations.to_string())
    
    # Statistics
    bets = recommendations[recommendations['bet_recommendation'] == 'BET']
    print(f"\n\nTotal matches: {len(recommendations)}")
    print(f"Recommended bets: {len(bets)}")
    print(f"Win rate: {len(bets) / len(recommendations) * 100:.1f}%")
    
    if len(bets) > 0:
        print(f"\nAverage edge: {bets['edge_pct'].mean():.2f}%")
        print(f"Average Kelly %: {bets['kelly_pct'].mean():.2f}%")
        print(f"Total potential points: {bets['bet_points_value'].sum()}")
        print(f"Average EV per bet: {bets['expected_value'].mean():.4f}")
        print(f"\nBet distribution by risk:")
        print(bets['risk_level'].value_counts().to_string())


if __name__ == '__main__':
    main()
