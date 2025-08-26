"""
Basketball Prediction Module
NBA and international basketball betting predictions with advanced ML
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from colorama import init, Fore, Back, Style

from config import CONFIG, get_sport_config
from utils import (ELORatingSystem, AdvancedFeatureEngineer, ModelTrainer, 
                  DataManager, WalkForwardValidator)

init(autoreset=True)
logger = logging.getLogger(__name__)

class BasketballPredictor:
    """Advanced Basketball Betting Predictor targeting 70%+ accuracy"""
    
    def __init__(self):
        self.sport = 'basketball'
        self.config = get_sport_config(self.sport)
        self.elo_system = ELORatingSystem(self.sport)
        self.feature_engineer = AdvancedFeatureEngineer(self.sport)
        self.model_trainer = ModelTrainer()
        self.data_manager = DataManager('basketball')
        
        # Load existing model if available
        self.model_data = self.model_trainer.load_model(self.sport)
        
        # Basketball leagues and competitions
        self.leagues = {
            'NBA': 'National Basketball Association',
            'EUROLEAGUE': 'EuroLeague',
            'NCAAB': 'NCAA Basketball',
            'WNBA': 'Women\'s NBA',
            'NBL': 'National Basketball League'
        }
        
        # NBA teams mapping
        self.nba_teams = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
            'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
        }
        
        print(f"{Fore.GREEN}🏀 Basketball Predictor initialized with advanced analytics{Style.RESET_ALL}")
    
    def fetch_basketball_data(self, league: str = 'NBA', 
                             season: str = '2024') -> Optional[pd.DataFrame]:
        """Fetch basketball data from multiple sources"""
        
        cache_key = f"basketball_{league}_{season}"
        
        # Try cache first
        cached_data = self.data_manager.get_cached_data(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {league} {season}")
            return pd.DataFrame(cached_data)
        
        # API endpoints vary by league
        if league == 'NBA':
            return self._fetch_nba_data(season)
        else:
            return self._fetch_generic_basketball_data(league, season)
    
    def _fetch_nba_data(self, season: str) -> Optional[pd.DataFrame]:
        """Fetch NBA data from API-Sports"""
        
        # API-Sports Basketball headers
        headers = {
            'X-RapidAPI-Host': 'v1.basketball.api-sports.io',
            'X-RapidAPI-Key': self.config['api_key']
        }
        url = f"{self.config['api_base']}/games"
        params = {'league': '12', 'season': f'{season}-{int(season)+1}'}  # NBA league ID is 12
        
        api_data = self.data_manager.fetch_api_data(url, headers, params, cache_key=f"nba_{season}")
        
        games_data = []
        
        if api_data and 'response' in api_data:
            for game in api_data['response']:
                if game['status']['short'] == 'FT':  # Finished games
                    game_data = {
                        'date': game['date'][:10],
                        'home_team': game['teams']['home']['name'],
                        'away_team': game['teams']['away']['name'],
                        'home_score': game['scores']['home']['total'],
                        'away_score': game['scores']['away']['total'],
                        'home_field_goals': 0,  # Would need separate stats API call
                        'away_field_goals': 0,
                        'home_three_pointers': 0,
                        'away_three_pointers': 0,
                        'home_rebounds': 0,
                        'away_rebounds': 0,
                        'home_assists': 0,
                        'away_assists': 0,
                        'home_turnovers': 0,
                        'away_turnovers': 0,
                        'league': 'NBA',
                        'season': season,
                        'game_id': game['id']
                    }
                    games_data.append(game_data)
        
        if games_data:
            df = pd.DataFrame(games_data)
            logger.info(f"Fetched {len(df)} NBA games")
            return df
        
        return None
    
    def _fetch_generic_basketball_data(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Fetch data for other basketball leagues"""
        
        # Simulate basketball data structure
        games_data = []
        
        # This would be replaced with actual API calls
        logger.info(f"Would fetch data for {league} {season}")
        
        return pd.DataFrame(games_data)
    
    def prepare_basketball_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare advanced basketball features"""
        df = df.copy()
        
        # Basic feature engineering
        df = self.feature_engineer.create_base_features(df)
        
        # Create match result for training
        df['result'] = df.apply(lambda x: 
            1 if x['home_score'] > x['away_score'] else -1, axis=1)
        
        # Basketball-specific features
        df = self._create_basketball_specific_features(df)
        
        # Team form features
        df = self.feature_engineer.create_team_form_features(
            df, 'home_team', 'result', window=10  # Longer window for basketball
        )
        df = self.feature_engineer.create_team_form_features(
            df, 'away_team', 'result', window=10
        )
        
        # Head-to-head features
        df = self.feature_engineer.create_head_to_head_features(
            df, 'home_team', 'away_team', 'result'
        )
        
        # ELO ratings
        df['home_elo'] = df['home_team'].apply(self.elo_system.get_rating)
        df['away_elo'] = df['away_team'].apply(self.elo_system.get_rating)
        df['elo_difference'] = df['home_elo'] - df['away_elo']
        
        # Update ELO ratings
        for idx, row in df.iterrows():
            self.elo_system.update_ratings(
                row['home_team'], 
                row['away_team'], 
                row['result'],
                home_advantage=True
            )
        
        return df
    
    def _create_basketball_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basketball-specific advanced features"""
        df = df.copy()
        
        # Scoring efficiency features
        if 'home_field_goals' in df.columns:
            # Field goal percentage (estimated if attempts not available)
            df['home_fg_pct'] = df['home_field_goals'] / (df['home_field_goals'] + 5)  # Estimated
            df['away_fg_pct'] = df['away_field_goals'] / (df['away_field_goals'] + 5)
            
            # Three-point reliance
            df['home_three_pct'] = df['home_three_pointers'] / (df['home_score'] / 2 + 1)
            df['away_three_pct'] = df['away_three_pointers'] / (df['away_score'] / 2 + 1)
        
        # Pace and efficiency
        df['total_score'] = df['home_score'] + df['away_score']
        df['score_difference'] = df['home_score'] - df['away_score']
        df['high_scoring_game'] = (df['total_score'] > 220).astype(int)
        df['blowout_game'] = (abs(df['score_difference']) > 20).astype(int)
        
        # Team strength indicators
        for team_type in ['home', 'away']:
            score_col = f'{team_type}_score'
            
            # Average points per game (rolling)
            df[f'{team_type}_ppg'] = df.groupby(f'{team_type}_team')[score_col].transform(
                lambda x: x.expanding().mean()
            )
            
            # Points allowed per game
            opp_score_col = 'away_score' if team_type == 'home' else 'home_score'
            df[f'{team_type}_papg'] = df.groupby(f'{team_type}_team')[opp_score_col].transform(
                lambda x: x.expanding().mean()
            )
            
            # Net rating (points differential per game)
            df[f'{team_type}_net_rating'] = df[f'{team_type}_ppg'] - df[f'{team_type}_papg']
            
            # Recent form (last 5 games)
            df[f'{team_type}_form_5'] = df.groupby(f'{team_type}_team')[score_col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
        
        # Offensive and defensive efficiency ratios
        df['home_off_efficiency'] = df['home_ppg'] / (df['away_papg'] + 80)  # League average baseline
        df['away_off_efficiency'] = df['away_ppg'] / (df['home_papg'] + 80)
        df['home_def_efficiency'] = df['away_papg'] / (df['home_ppg'] + 80)
        df['away_def_efficiency'] = df['home_papg'] / (df['away_ppg'] + 80)
        
        # Advanced metrics if available
        if 'home_rebounds' in df.columns:
            # Rebounding advantage
            df['rebounding_advantage'] = (df['home_rebounds'] - df['away_rebounds']) / df['total_score']
            
            # Assist to turnover ratio
            df['home_ast_to_ratio'] = (df['home_assists'] + 1) / (df['home_turnovers'] + 1)
            df['away_ast_to_ratio'] = (df['away_assists'] + 1) / (df['away_turnovers'] + 1)
        
        # Rest days and back-to-back games
        df['date'] = pd.to_datetime(df['date'])
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            
            # Days since last game
            df[f'{team_type}_rest_days'] = df.groupby(team_col)['date'].diff().dt.days.fillna(3)
            
            # Back-to-back games indicator
            df[f'{team_type}_back_to_back'] = (df[f'{team_type}_rest_days'] <= 1).astype(int)
        
        # Season context
        df['games_played'] = df.groupby(['home_team', 'away_team']).cumcount() + 1
        df['season_progress'] = df['games_played'] / 82  # NBA season length
        
        return df
    
    def train_basketball_model(self, df: pd.DataFrame) -> Dict:
        """Train enhanced basketball prediction model"""
        
        print(f"{Fore.CYAN}🔧 Training basketball model with advanced analytics...{Style.RESET_ALL}")
        
        # Prepare features
        feature_df = self.prepare_basketball_features(df)
        
        # Select features for training
        exclude_cols = [
            'date', 'home_team', 'away_team', 'home_score', 'away_score',
            'game_id', 'season', 'result', 'league'
        ]
        
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = feature_df[feature_cols].fillna(0)
        y = (feature_df['result'] + 1) // 2  # Convert -1,1 to 0,1 for binary classification
        
        # Train ensemble model
        model_result = self.model_trainer.train_ensemble_model(X, y, self.sport)
        
        self.model_data = model_result
        
        print(f"{Fore.GREEN}✅ Basketball model trained with accuracy: {model_result['best_score']:.3f}{Style.RESET_ALL}")
        
        return model_result
    
    def predict_game(self, home_team: str, away_team: str, 
                    league: str = 'NBA') -> Dict[str, float]:
        """Predict basketball game outcome"""
        
        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model available. Please train first.{Style.RESET_ALL}")
            return {}
        
        # Create feature vector for prediction
        game_features = self._create_game_features(home_team, away_team, league)
        
        if game_features is None:
            return {}
        
        # Get model prediction
        model = self.model_data['best_model']
        
        # Scale and transform features
        scaler = self.model_trainer.scalers.get('scaler')
        if scaler:
            game_features_scaled = scaler.transform([game_features])
        else:
            game_features_scaled = [game_features]
        
        feature_selector = self.model_trainer.feature_selectors.get('selector')
        if feature_selector:
            game_features_scaled = feature_selector.transform(game_features_scaled)
        
        # Get probabilities
        probabilities = model.predict_proba(game_features_scaled)[0]
        
        # Get ELO-based probabilities
        elo_probs = self.elo_system.get_match_probability(
            home_team, away_team, home_advantage=True
        )
        
        # Combine predictions
        model_weight = 0.75  # Basketball models tend to be more reliable
        elo_weight = 0.25
        
        final_probs = {
            'home_win': model_weight * probabilities[1] + elo_weight * elo_probs[0],
            'away_win': model_weight * probabilities[0] + elo_weight * elo_probs[2]
        }
        
        # Add point spread prediction (simplified)
        elo_diff = self.elo_system.get_rating(home_team) - self.elo_system.get_rating(away_team)
        predicted_spread = elo_diff / 25  # Rough conversion
        final_probs['predicted_spread'] = predicted_spread
        
        # Add total points prediction (simplified)
        avg_total = 220  # NBA average
        final_probs['predicted_total'] = avg_total
        
        # Confidence and recommendation
        confidence = abs(final_probs['home_win'] - final_probs['away_win'])
        final_probs['confidence'] = confidence
        
        if confidence > CONFIG['ml'].min_confidence_threshold:
            best_outcome = 'home_win' if final_probs['home_win'] > final_probs['away_win'] else 'away_win'
            final_probs['recommendation'] = best_outcome
        else:
            final_probs['recommendation'] = 'no_bet'
        
        return final_probs
    
    def _create_game_features(self, home_team: str, away_team: str, 
                             league: str) -> Optional[List[float]]:
        """Create feature vector for game prediction"""
        
        try:
            home_elo = self.elo_system.get_rating(home_team)
            away_elo = self.elo_system.get_rating(away_team)
            
            # Basic features for basketball
            features = [
                home_elo,
                away_elo,
                home_elo - away_elo,
                1,  # Home court advantage
                datetime.now().month,
                datetime.now().weekday(),
                0,  # Recent form (would be calculated)
                0,  # Points per game
                0,  # Points allowed per game
                0,  # Net rating
                0,  # Rest days
                0,  # Back-to-back indicator
                0.5  # Season progress
            ]
            
            # Pad to match training features
            expected_features = len(self.model_data.get('feature_names', []))
            if expected_features > 0:
                if len(features) < expected_features:
                    features.extend([0] * (expected_features - len(features)))
                elif len(features) > expected_features:
                    features = features[:expected_features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating game features: {e}")
            return None
    
    def get_upcoming_games(self, league: str = 'NBA') -> List[Dict]:
        """Get upcoming basketball games"""
        
        cache_key = f"upcoming_basketball_{league}"
        
        if league == 'NBA':
            # NBA API call for upcoming games
            headers = {'Ocp-Apim-Subscription-Key': self.config['api_key']}
            today = datetime.now().strftime('%Y-%m-%d')
            url = f"{self.config['api_base']}/scores/json/GamesByDate/{today}"
            
            api_data = self.data_manager.fetch_api_data(url, headers, cache_key=cache_key)
            
            upcoming_games = []
            
            if api_data:
                for game in api_data[:10]:  # Limit to next 10 games
                    if game.get('Status') in ['Scheduled', 'InProgress']:
                        game_info = {
                            'date': game['Day'][:10],
                            'time': game.get('DateTime', '')[-8:-3],
                            'home_team': game['HomeTeam'],
                            'away_team': game['AwayTeam'],
                            'league': league,
                            'game_id': game['GameID']
                        }
                        upcoming_games.append(game_info)
            
            return upcoming_games
        
        return []
    
    def analyze_upcoming_games(self, league: str = 'NBA') -> List[Dict]:
        """Analyze upcoming games and provide recommendations"""
        
        upcoming_games = self.get_upcoming_games(league)
        recommendations = []
        
        print(f"{Fore.CYAN}🔍 Analyzing upcoming {self.leagues.get(league, league)} games...{Style.RESET_ALL}")
        
        for game in upcoming_games:
            prediction = self.predict_game(
                game['home_team'],
                game['away_team'],
                league
            )
            
            if prediction:
                recommendation = {
                    'game': f"{game['home_team']} vs {game['away_team']}",
                    'date': game['date'],
                    'time': game['time'],
                    'league': self.leagues.get(league, league),
                    'predictions': prediction,
                    'confidence': prediction.get('confidence', 0),
                    'recommended_bet': prediction.get('recommendation', 'no_bet')
                }
                recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations
    
    def display_predictions(self, recommendations: List[Dict]):
        """Display basketball predictions"""
        
        if not recommendations:
            print(f"{Fore.YELLOW}⚠️ No betting opportunities found{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🏀 BASKETBALL BETTING PREDICTIONS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'='*60}{Style.RESET_ALL}")
        
        for i, rec in enumerate(recommendations, 1):
            if rec['recommended_bet'] != 'no_bet':
                print(f"\n{Fore.CYAN}{Style.BRIGHT}{i}. {rec['game']}{Style.RESET_ALL}")
                print(f"   📅 {rec['date']} {rec['time']} | 🏆 {rec['league']}")
                print(f"   🎯 Recommendation: {Fore.GREEN}{rec['recommended_bet'].replace('_', ' ').title()}{Style.RESET_ALL}")
                print(f"   📊 Confidence: {Fore.YELLOW}{rec['confidence']:.1%}{Style.RESET_ALL}")
                
                preds = rec['predictions']
                print(f"   📈 Probabilities:")
                print(f"      🏠 Home Win: {preds.get('home_win', 0):.1%}")
                print(f"      ✈️ Away Win: {preds.get('away_win', 0):.1%}")
                
                if 'predicted_spread' in preds:
                    print(f"      📏 Predicted Spread: {preds['predicted_spread']:+.1f}")
                if 'predicted_total' in preds:
                    print(f"      🎯 Predicted Total: {preds['predicted_total']:.1f}")
    
    def run(self):
        """Main run method for basketball predictor"""
        
        while True:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}🏀 BASKETBALL PREDICTOR{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
            print(f"1. 🎓 Train Model")
            print(f"2. 🔮 Make Predictions")
            print(f"3. 📊 Model Performance")
            print(f"4. ⚙️ Settings")
            print(f"5. 🔙 Return to Main Menu")
            
            try:
                choice = input(f"\n{Fore.YELLOW}Select option: {Style.RESET_ALL}").strip()
                
                if choice == '1':
                    self.run_training_mode()
                elif choice == '2':
                    self.run_prediction_mode()
                elif choice == '3':
                    self.show_model_performance()
                elif choice == '4':
                    self.show_settings()
                elif choice == '5':
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please select 1-5.{Style.RESET_ALL}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def run_training_mode(self):
        """Run training mode for basketball"""
        
        print(f"{Fore.YELLOW}🎓 Basketball Training Mode{Style.RESET_ALL}")
        
        # For demo purposes, create sample data
        print("Fetching NBA season data...")
        
        # This would fetch real data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'home_team': np.random.choice(list(self.nba_teams.values()), 100),
            'away_team': np.random.choice(list(self.nba_teams.values()), 100),
            'home_score': np.random.randint(90, 130, 100),
            'away_score': np.random.randint(90, 130, 100)
        })
        
        # Remove games where team plays itself
        sample_data = sample_data[sample_data['home_team'] != sample_data['away_team']]
        
        if len(sample_data) > 0:
            print(f"Training on {len(sample_data)} games")
            self.train_basketball_model(sample_data)
        else:
            print(f"{Fore.RED}❌ No training data available{Style.RESET_ALL}")
    
    def run_prediction_mode(self):
        """Run prediction mode for basketball"""
        
        print(f"{Fore.YELLOW}🔮 Basketball Prediction Mode{Style.RESET_ALL}")
        
        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model found. Please run training first.{Style.RESET_ALL}")
            return
        
        recommendations = self.analyze_upcoming_games('NBA')
        self.display_predictions(recommendations)
    
    def show_model_performance(self):
        """Display model performance metrics"""
        
        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model available{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}📊 BASKETBALL MODEL PERFORMANCE{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
        print(f"Best Model: {self.model_data['best_model_name']}")
        print(f"Accuracy: {self.model_data['best_score']:.1%}")
        
        print(f"\nAll Model Scores:")
        for model_name, score in self.model_data['model_scores'].items():
            print(f"  {model_name}: {score:.1%}")
    
    def show_settings(self):
        """Display current basketball settings"""
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}⚙️ BASKETBALL SETTINGS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
        print(f"Target Accuracy: {CONFIG['ml'].target_accuracy:.1%}")
        print(f"Confidence Threshold: {CONFIG['ml'].min_confidence_threshold:.1%}")
        print(f"ELO K-Factor: {CONFIG['elo'].basketball_k_factor}")
        print(f"Available Markets: {', '.join(CONFIG['betting'].basketball_markets)}")

if __name__ == "__main__":
    predictor = BasketballPredictor()
    predictor.run()
