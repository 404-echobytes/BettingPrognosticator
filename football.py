"""
Enhanced Football Prediction Module
Building upon original betfing-predictor with advanced ML pipeline
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

class FootballPredictor:
    """Enhanced Football Betting Predictor with 70%+ accuracy target"""

    def __init__(self):
        self.sport = 'football'
        self.config = get_sport_config(self.sport)
        self.elo_system = ELORatingSystem(self.sport)
        self.feature_engineer = AdvancedFeatureEngineer(self.sport)
        self.model_trainer = ModelTrainer()
        self.data_manager = DataManager()

        # Load existing model if available
        self.model_data = self.model_trainer.load_model(self.sport)

        # Football-specific leagues and competitions
        self.leagues = {
            'PL': 'Premier League',
            'PD': 'La Liga', 
            'BL1': 'Bundesliga',
            'SA': 'Serie A',
            'FL1': 'Ligue 1',
            'DED': 'Eredivisie',
            'PPL': 'Primeira Liga',
            'CLI': 'Champions League',
            'ELC': 'Europa League'
        }

        print(f"{Fore.GREEN}⚽ Football Predictor initialized with enhanced ML pipeline{Style.RESET_ALL}")

    def fetch_football_data(self, league: str = 'PL', 
                           season: str = '2024') -> Optional[pd.DataFrame]:
        """Fetch football data from API with enhanced caching"""

        cache_key = f"football_{league}_{season}"

        # Try to get from cache first
        cached_data = self.data_manager.get_cached_data(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {league} {season}")
            return pd.DataFrame(cached_data)

        # Fetch from API
        headers = {'X-Auth-Token': self.config['api_key']}
        url = f"{self.config['api_base']}/competitions/{league}/matches"
        params = {'season': season}

        api_data = self.data_manager.fetch_api_data(url, headers, params, cache_key)

        matches_data = []
        if api_data and 'response' in api_data:
            for fixture in api_data['response']:
                if fixture['fixture']['status']['short'] == 'FT':
                    match_data = {
                        'date': fixture['fixture']['date'][:10],
                        'home_team': fixture['teams']['home']['name'],
                        'away_team': fixture['teams']['away']['name'],
                        'home_score': fixture['goals']['home'],
                        'away_score': fixture['goals']['away'],
                        'league': league,
                        'season': season,
                        'match_id': fixture['fixture']['id']
                    }
                    matches_data.append(match_data)

        if matches_data:
            df = pd.DataFrame(matches_data)
            logger.info(f"Fetched {len(df)} matches for {league} {season}")
            return df

        logger.warning(f"No data available for {league} {season}")
        return None

    def prepare_football_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare advanced football features"""
        df = df.copy()

        # Basic feature engineering
        df = self.feature_engineer.create_base_features(df)

        # Create match result for training
        df['result'] = df.apply(lambda x: 
            1 if x['home_score'] > x['away_score'] else 
            (-1 if x['home_score'] < x['away_score'] else 0), axis=1)

        # Create form features for both teams
        df = self.feature_engineer.create_team_form_features(
            df, 'home_team', 'result', window=5
        )
        df = self.feature_engineer.create_team_form_features(
            df, 'away_team', 'result', window=5
        )

        # Head-to-head features
        df = self.feature_engineer.create_head_to_head_features(
            df, 'home_team', 'away_team', 'result'
        )

        # Football-specific features
        df = self.feature_engineer.create_sport_specific_features(df)

        # ELO ratings
        df['home_elo'] = df['home_team'].apply(self.elo_system.get_rating)
        df['away_elo'] = df['away_team'].apply(self.elo_system.get_rating)
        df['elo_difference'] = df['home_elo'] - df['away_elo']

        # Update ELO ratings based on results
        for idx, row in df.iterrows():
            result_for_elo = 1 if row['result'] == 1 else (-1 if row['result'] == -1 else 0)
            self.elo_system.update_ratings(
                row['home_team'], 
                row['away_team'], 
                result_for_elo,
                home_advantage=True
            )

        # Additional advanced features
        df = self._create_advanced_football_features(df)

        return df

    def _create_advanced_football_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced football-specific features"""
        df = df.copy()

        # Goal scoring features
        df['home_goals_per_game'] = df.groupby('home_team')['home_score'].transform(
            lambda x: x.expanding().mean()
        )
        df['away_goals_per_game'] = df.groupby('away_team')['away_score'].transform(
            lambda x: x.expanding().mean()
        )
        df['home_goals_conceded_per_game'] = df.groupby('home_team')['away_score'].transform(
            lambda x: x.expanding().mean()
        )
        df['away_goals_conceded_per_game'] = df.groupby('away_team')['home_score'].transform(
            lambda x: x.expanding().mean()
        )

        # Attack vs Defense strength
        df['home_attack_strength'] = df['home_goals_per_game'] / (df['away_goals_conceded_per_game'] + 0.1)
        df['away_attack_strength'] = df['away_goals_per_game'] / (df['home_goals_conceded_per_game'] + 0.1)

        # Recent form (last 3 games)
        for team_type in ['home', 'home']:
            team_col = f'{team_type}_team'
            score_col = f'{team_type}_score'

            # Goals in last 3 games
            df[f'{team_type}_goals_l3'] = df.groupby(team_col)[score_col].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )

            # Clean sheets in last 5 games (for defense)
            opp_score_col = 'away_score' if team_type == 'home' else 'home_score'
            df[f'{team_type}_clean_sheets_l5'] = df.groupby(team_col)[opp_score_col].transform(
                lambda x: (x.rolling(window=5, min_periods=1) == 0).sum()
            )

        # Market-specific features
        df['btts_probability'] = ((df['home_goals_per_game'] > 0.8) & 
                                 (df['away_goals_per_game'] > 0.8)).astype(float)

        df['over_2_5_probability'] = ((df['home_goals_per_game'] + df['away_goals_per_game']) > 2.0).astype(float)

        # League strength adjustment
        if 'league' in df.columns:
            df['total_goals'] = df['home_score'] + df['away_score']
            league_goal_avg = df.groupby('league')['total_goals'].transform('mean')
            df['league_adjusted_goals'] = df['total_goals'] / (league_goal_avg + 0.1)

        return df

    def train_football_model(self, df: pd.DataFrame) -> Dict:
        """Train enhanced football prediction model"""

        print(f"{Fore.CYAN}🔧 Training enhanced football model...{Style.RESET_ALL}")

        # Prepare features
        feature_df = self.prepare_football_features(df)

        # Select features for training (excluding non-predictive columns)
        exclude_cols = [
            'date', 'home_team', 'away_team', 'home_score', 'away_score', 
            'match_id', 'season', 'result', 'match_key', 'last_meeting'
        ]

        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]

        # Handle missing values
        X = feature_df[feature_cols].fillna(0)
        y = feature_df['result'] + 1  # Convert to 0, 1, 2 for away win, draw, home win

        # Train ensemble model
        model_result = self.model_trainer.train_ensemble_model(X, y, self.sport)

        self.model_data = model_result

        print(f"{Fore.GREEN}✅ Football model trained with accuracy: {model_result['best_score']:.3f}{Style.RESET_ALL}")

        return model_result

    def predict_match(self, home_team: str, away_team: str, 
                     league: str = 'PL') -> Dict[str, float]:
        """Predict match outcome with confidence scores"""

        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model available. Please train first.{Style.RESET_ALL}")
            return {}

        # Create feature vector for prediction
        match_features = self._create_match_features(home_team, away_team, league)

        if match_features is None:
            print(f"{Fore.RED}❌ Could not create features for prediction{Style.RESET_ALL}")
            return {}

        # Get model prediction
        model = self.model_data['best_model']

        # Scale features using saved scaler
        scaler = self.model_trainer.scalers.get('scaler')
        if scaler:
            match_features_scaled = scaler.transform([match_features])
        else:
            match_features_scaled = [match_features]

        # Apply feature selection if used
        feature_selector = self.model_trainer.feature_selectors.get('selector')
        if feature_selector:
            match_features_scaled = feature_selector.transform(match_features_scaled)

        # Get probabilities
        probabilities = model.predict_proba(match_features_scaled)[0]

        # Get ELO-based probabilities for comparison
        elo_probs = self.elo_system.get_match_probability(
            home_team, away_team, home_advantage=True
        )

        # Combine model and ELO predictions (weighted average)
        model_weight = 0.7
        elo_weight = 0.3

        final_probs = {
            'home_win': model_weight * probabilities[2] + elo_weight * elo_probs[0],
            'draw': model_weight * probabilities[1] + elo_weight * elo_probs[1],
            'away_win': model_weight * probabilities[0] + elo_weight * elo_probs[2]
        }

        # Add confidence score
        confidence = max(final_probs.values()) - (sum(final_probs.values()) - max(final_probs.values())) / 2
        final_probs['confidence'] = confidence

        # Add recommended bet if confidence is high enough
        if confidence > CONFIG['ml'].min_confidence_threshold:
            best_outcome = max(final_probs, key=lambda k: final_probs[k] if k != 'confidence' else 0)
            final_probs['recommendation'] = best_outcome
        else:
            final_probs['recommendation'] = 'no_bet'

        return final_probs

    def _create_match_features(self, home_team: str, away_team: str, 
                              league: str) -> Optional[List[float]]:
        """Create feature vector for a single match prediction"""

        # This would normally use recent team data to create features
        # For now, we'll use ELO ratings and basic features
        try:
            home_elo = self.elo_system.get_rating(home_team)
            away_elo = self.elo_system.get_rating(away_team)

            # Basic features (this would be expanded with real data)
            features = [
                home_elo,
                away_elo,
                home_elo - away_elo,  # ELO difference
                1,  # Home advantage
                datetime.now().month,  # Current month
                datetime.now().weekday(),  # Day of week
                0,  # Recent form (would be calculated from recent matches)
                0,  # H2H record
                0,  # Goals per game
                0   # Clean sheets
            ]

            # Pad or truncate to match training features
            expected_features = len(self.model_data.get('feature_names', []))
            if expected_features > 0:
                if len(features) < expected_features:
                    features.extend([0] * (expected_features - len(features)))
                elif len(features) > expected_features:
                    features = features[:expected_features]

            return features

        except Exception as e:
            logger.error(f"Error creating match features: {e}")
            return None

    def get_upcoming_matches(self, league: str = 'PL') -> List[Dict]:
        """Get upcoming matches for prediction"""

        cache_key = f"upcoming_football_{league}"

        headers = {'X-Auth-Token': self.config['api_key']}
        url = f"{self.config['api_base']}/competitions/{league}/matches"
        params = {'status': 'SCHEDULED'}

        api_data = self.data_manager.fetch_api_data(url, headers, params, cache_key)

        upcoming_matches = []

        if api_data and 'response' in api_data:
            for fixture in api_data['response'][:10]:  # Limit to next 10 matches
                match_info = {
                    'date': fixture['fixture']['date'][:10],
                    'time': fixture['fixture']['date'][11:16],
                    'home_team': fixture['teams']['home']['name'],
                    'away_team': fixture['teams']['away']['name'],
                    'league': league,
                    'match_id': fixture['fixture']['id']
                }
                upcoming_matches.append(match_info)

        return upcoming_matches

    def analyze_upcoming_matches(self, league: str = 'PL') -> List[Dict]:
        """Analyze upcoming matches and provide betting recommendations"""

        upcoming_matches = self.get_upcoming_matches(league)
        recommendations = []

        print(f"{Fore.CYAN}🔍 Analyzing upcoming {self.leagues.get(league, league)} matches...{Style.RESET_ALL}")

        for match in upcoming_matches:
            prediction = self.predict_match(
                match['home_team'], 
                match['away_team'], 
                league
            )

            if prediction:
                recommendation = {
                    'match': f"{match['home_team']} vs {match['away_team']}",
                    'date': match['date'],
                    'time': match['time'],
                    'league': self.leagues.get(league, league),
                    'predictions': prediction,
                    'confidence': prediction.get('confidence', 0),
                    'recommended_bet': prediction.get('recommendation', 'no_bet')
                }
                recommendations.append(recommendation)

        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)

        return recommendations

    def display_predictions(self, recommendations: List[Dict]):
        """Display predictions in a formatted way"""

        if not recommendations:
            print(f"{Fore.YELLOW}⚠️ No betting opportunities found{Style.RESET_ALL}")
            return

        print(f"\n{Fore.GREEN}{Style.BRIGHT}⚽ FOOTBALL BETTING PREDICTIONS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'='*60}{Style.RESET_ALL}")

        for i, rec in enumerate(recommendations, 1):
            if rec['recommended_bet'] != 'no_bet':
                print(f"\n{Fore.CYAN}{Style.BRIGHT}{i}. {rec['match']}{Style.RESET_ALL}")
                print(f"   📅 {rec['date']} {rec['time']} | 🏆 {rec['league']}")
                print(f"   🎯 Recommendation: {Fore.GREEN}{rec['recommended_bet'].replace('_', ' ').title()}{Style.RESET_ALL}")
                print(f"   📊 Confidence: {Fore.YELLOW}{rec['confidence']:.1%}{Style.RESET_ALL}")

                preds = rec['predictions']
                print(f"   📈 Probabilities:")
                print(f"      🏠 Home Win: {preds.get('home_win', 0):.1%}")
                print(f"      🤝 Draw: {preds.get('draw', 0):.1%}")
                print(f"      ✈️ Away Win: {preds.get('away_win', 0):.1%}")

    def run_training_mode(self):
        """Run training mode with multiple leagues"""

        print(f"{Fore.YELLOW}🎓 Football Training Mode{Style.RESET_ALL}")
        print(f"Available leagues: {', '.join(self.leagues.keys())}")

        # Collect data from multiple leagues
        all_data = []

        for league_code in self.leagues.keys():
            print(f"Fetching data for {self.leagues[league_code]}...")

            # Get current and previous season
            current_year = datetime.now().year
            for year in [current_year, current_year - 1]:
                season_data = self.fetch_football_data(league_code, str(year))
                if season_data is not None:
                    all_data.append(season_data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Training on {len(combined_data)} matches")

            # Train model
            self.train_football_model(combined_data)
        else:
            print(f"{Fore.RED}❌ No training data available{Style.RESET_ALL}")

    def run_prediction_mode(self):
        """Run prediction mode"""

        print(f"{Fore.YELLOW}🔮 Football Prediction Mode{Style.RESET_ALL}")

        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model found. Please run training first.{Style.RESET_ALL}")
            return

        while True:
            print(f"\n{Fore.CYAN}Select league for predictions:{Style.RESET_ALL}")
            for i, (code, name) in enumerate(self.leagues.items(), 1):
                print(f"{i}. {name} ({code})")
            print(f"{len(self.leagues) + 1}. Return to main menu")

            try:
                choice = int(input(f"{Fore.YELLOW}Enter choice: {Style.RESET_ALL}"))

                if choice == len(self.leagues) + 1:
                    break
                elif 1 <= choice <= len(self.leagues):
                    league_code = list(self.leagues.keys())[choice - 1]
                    recommendations = self.analyze_upcoming_matches(league_code)
                    self.display_predictions(recommendations)
                else:
                    print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")

            except ValueError:
                print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
            except KeyboardInterrupt:
                break

    def run(self):
        """Main run method for football predictor"""

        while True:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}⚽ FOOTBALL PREDICTOR{Style.RESET_ALL}")
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

    def show_model_performance(self):
        """Display model performance metrics"""

        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model available{Style.RESET_ALL}")
            return

        print(f"\n{Fore.GREEN}{Style.BRIGHT}📊 MODEL PERFORMANCE{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
        print(f"Best Model: {self.model_data['best_model_name']}")
        print(f"Accuracy: {self.model_data['best_score']:.1%}")

        print(f"\nAll Model Scores:")
        for model_name, score in self.model_data['model_scores'].items():
            print(f"  {model_name}: {score:.1%}")

    def show_settings(self):
        """Display current settings"""

        print(f"\n{Fore.GREEN}{Style.BRIGHT}⚙️ FOOTBALL SETTINGS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
        print(f"Target Accuracy: {CONFIG['ml'].target_accuracy:.1%}")
        print(f"Confidence Threshold: {CONFIG['ml'].min_confidence_threshold:.1%}")
        print(f"ELO K-Factor: {CONFIG['elo'].football_k_factor}")
        print(f"Available Markets: {', '.join(CONFIG['betting'].football_markets)}")

if __name__ == "__main__":
    predictor = FootballPredictor()
    predictor.run()