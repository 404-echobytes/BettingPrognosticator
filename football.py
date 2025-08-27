"""
Tennis Prediction Module
ATP/WTA tennis betting predictions with surface analysis and player ranking features
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

class TennisPredictor:
    """Advanced Tennis Betting Predictor with surface analysis targeting 70%+ accuracy"""
    
    def __init__(self):
        self.sport = 'tennis'
        self.config = get_sport_config(self.sport)
        self.elo_system = ELORatingSystem(self.sport)
        self.feature_engineer = AdvancedFeatureEngineer(self.sport)
        self.model_trainer = ModelTrainer()
        self.data_manager = DataManager('tennis')
        
        # Load existing model if available
        self.model_data = self.model_trainer.load_model(self.sport)
        
        # Tennis tours and surfaces
        self.tours = {
            'ATP': 'Association of Tennis Professionals',
            'WTA': 'Women\'s Tennis Association',
            'CHALLENGER': 'ATP Challenger Tour',
            'ITF': 'International Tennis Federation'
        }
        
        self.surfaces = {
            'hard': 'Hard Court',
            'clay': 'Clay Court', 
            'grass': 'Grass Court',
            'carpet': 'Carpet Court'
        }
        
        # Major tournaments
        self.grand_slams = {
            'australian_open': 'Australian Open',
            'french_open': 'Roland Garros',
            'wimbledon': 'Wimbledon',
            'us_open': 'US Open'
        }
        
        self.masters_1000 = {
            'indian_wells': 'BNP Paribas Open',
            'miami': 'Miami Open',
            'monte_carlo': 'Monte-Carlo Masters',
            'madrid': 'Madrid Open',
            'rome': 'Italian Open',
            'canada': 'Canadian Open',
            'cincinnati': 'Cincinnati Masters',
            'shanghai': 'Shanghai Masters',
            'paris': 'Paris Masters'
        }
        
        print(f"{Fore.GREEN}🎾 Tennis Predictor initialized with surface analysis{Style.RESET_ALL}")
    
    def fetch_tennis_data(self, tour: str = 'ATP', 
                         year: str = '2024') -> Optional[pd.DataFrame]:
        """Fetch tennis data from RapidAPI (free version)"""
        
        cache_key = f"tennis_{tour}_{year}"
        
        # Try cache first
        cached_data = self.data_manager.get_cached_data(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {tour} {year}")
            return pd.DataFrame(cached_data)
        
        # RapidAPI headers for tennis
        headers = {
            'X-RapidAPI-Host': 'ultimate-tennis1.p.rapidapi.com',
            'X-RapidAPI-Key': self.config['api_key']
        }
        
        # Adjust URL for free RapidAPI version
        url = f"{self.config['api_base']}/tournaments_matches"
        params = {'tour': tour.lower(), 'year': year}
        
        api_data = self.data_manager.fetch_api_data(url, headers, params, cache_key)
        
        matches_data = []
        
        if api_data and 'matches' in api_data:
            for match in api_data['matches']:
                if match.get('status') == 'completed':
                    match_data = {
                        'date': match['start_date'][:10],
                        'player1': match['competitors'][0]['name'],
                        'player2': match['competitors'][1]['name'],
                        'player1_rank': int(match['competitors'][0].get('rank', 999)),
                        'player2_rank': int(match['competitors'][1].get('rank', 999)),
                        'winner': match['winner']['name'],
                        'sets_won_1': int(match['competitors'][0].get('sets_won', 0)),
                        'sets_won_2': int(match['competitors'][1].get('sets_won', 0)),
                        'games_won_1': int(match['competitors'][0].get('games_won', 0)),
                        'games_won_2': int(match['competitors'][1].get('games_won', 0)),
                        'surface': match.get('surface', 'hard'),
                        'tournament': match.get('tournament_name', 'Unknown'),
                        'round': match.get('round', 'R1'),
                        'tour': tour,
                        'year': year,
                        'match_id': str(match['id'])
                    }
                    matches_data.append(match_data)
        
        # Fallback to sample data if API fails
        if not matches_data:
            logger.warning(f"API failed, generating sample tennis data for {tour} {year}")
            return self._generate_sample_tennis_data(tour, year)
        
        df = pd.DataFrame(matches_data)
        logger.info(f"Fetched {len(df)} {tour} matches for {year}")
        return df
    
    def _generate_sample_tennis_data(self, tour: str, year: str) -> pd.DataFrame:
        """Generate sample tennis data for testing"""
        np.random.seed(42)
        
        # Sample players by tour
        players_by_tour = {
            'ATP': ['Novak Djokovic', 'Rafael Nadal', 'Roger Federer', 'Carlos Alcaraz', 'Daniil Medvedev', 
                   'Alexander Zverev', 'Stefanos Tsitsipas', 'Andrey Rublev', 'Casper Ruud', 'Jannik Sinner'],
            'WTA': ['Iga Swiatek', 'Aryna Sabalenka', 'Jessica Pegula', 'Elena Rybakina', 'Caroline Garcia',
                   'Ons Jabeur', 'Simona Halep', 'Coco Gauff', 'Maria Sakkari', 'Petra Kvitova']
        }
        
        players = players_by_tour.get(tour, players_by_tour['ATP'])
        surfaces = ['hard', 'clay', 'grass']
        
        matches = []
        start_date = datetime.strptime(f"{year}-01-01", '%Y-%m-%d')
        
        for i in range(150):
            player1 = np.random.choice(players)
            player2 = np.random.choice([p for p in players if p != player1])
            surface = np.random.choice(surfaces)
            
            # Generate match result
            winner = np.random.choice([player1, player2])
            if winner == player1:
                sets_1, sets_2 = np.random.choice([[2, 0], [2, 1]]), np.random.choice([[0, 2], [1, 2]])
                sets_won_1, sets_won_2 = sets_1, sets_2
            else:
                sets_1, sets_2 = np.random.choice([[0, 2], [1, 2]]), np.random.choice([[2, 0], [2, 1]])
                sets_won_1, sets_won_2 = sets_1, sets_2
            
            match_date = start_date + timedelta(days=i*2)
            
            matches.append({
                'date': match_date.strftime('%Y-%m-%d'),
                'player1': player1,
                'player2': player2,
                'player1_rank': np.random.randint(1, 100),
                'player2_rank': np.random.randint(1, 100),
                'winner': winner,
                'sets_won_1': sets_won_1,
                'sets_won_2': sets_won_2,
                'games_won_1': np.random.randint(10, 25),
                'games_won_2': np.random.randint(10, 25),
                'surface': surface,
                'tournament': f'Tournament_{i%10}',
                'round': np.random.choice(['R1', 'R2', 'R3', 'QF', 'SF', 'F']),
                'tour': tour,
                'year': year,
                'match_id': f"sample_{i}"
            })
        
        return pd.DataFrame(matches)
    
    def prepare_tennis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare advanced tennis features with surface analysis"""
        df = df.copy()
        
        # Basic feature engineering
        df = self.feature_engineer.create_base_features(df)
        
        # Create match result for training (1 if player1 wins, 0 if player2 wins)
        df['result'] = (df['winner'] == df['player1']).astype(int)
        
        # Tennis-specific features
        df = self._create_tennis_specific_features(df)
        
        # Player form features
        df = self._create_player_form_features(df)
        
        # Head-to-head features
        df = self.feature_engineer.create_head_to_head_features(
            df, 'player1', 'player2', 'result'
        )
        
        # Surface-specific features
        df = self._create_surface_features(df)
        
        # Ranking features
        df = self._create_ranking_features(df)
        
        # ELO ratings (overall and surface-specific)
        df['player1_elo'] = df['player1'].apply(self.elo_system.get_rating)
        df['player2_elo'] = df['player2'].apply(self.elo_system.get_rating)
        df['elo_difference'] = df['player1_elo'] - df['player2_elo']
        
        # Update ELO ratings
        for idx, row in df.iterrows():
            result_for_elo = 1 if row['result'] == 1 else -1
            self.elo_system.update_ratings(
                row['player1'], 
                row['player2'], 
                result_for_elo,
                home_advantage=False  # No home advantage in tennis
            )
        
        return df
    
    def _create_tennis_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tennis-specific features"""
        df = df.copy()
        
        # Match length indicators
        df['total_sets'] = df['sets_won_1'] + df['sets_won_2']
        df['total_games'] = df['games_won_1'] + df['games_won_2']
        df['games_difference'] = df['games_won_1'] - df['games_won_2']
        
        # Match competitiveness
        df['close_match'] = (abs(df['sets_won_1'] - df['sets_won_2']) <= 1).astype(int)
        df['straight_sets'] = ((df['sets_won_1'] == 2) | (df['sets_won_2'] == 2)).astype(int)
        
        # Tournament importance
        df['is_grand_slam'] = df['tournament'].str.lower().isin([
            'australian open', 'french open', 'wimbledon', 'us open'
        ]).astype(int)
        
        df['is_masters'] = df['tournament'].str.lower().str.contains(
            'masters|indian wells|miami|madrid|rome|canada|cincinnati|shanghai|paris'
        ).astype(int)
        
        # Round importance (later rounds = higher importance)
        round_importance = {
            'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4,
            'QF': 5, 'SF': 6, 'F': 7
        }
        df['round_importance'] = df['round'].map(round_importance).fillna(1)
        
        return df
    
    def _create_player_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create player form and performance features"""
        df = df.copy()
        
        # Sort by date for time-based features
        df = df.sort_values('date')
        
        for player_num in [1, 2]:
            player_col = f'player{player_num}'
            sets_col = f'sets_won_{player_num}'
            games_col = f'games_won_{player_num}'
            
            # Recent form (last 10 matches)
            df[f'player{player_num}_form_10'] = df.groupby(player_col)['result'].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean() if player_num == 1 
                else (1 - x).rolling(window=10, min_periods=1).mean()
            )
            
            # Win percentage (expanding window)
            df[f'player{player_num}_win_pct'] = df.groupby(player_col)['result'].transform(
                lambda x: x.expanding().mean() if player_num == 1
                else (1 - x).expanding().mean()
            )
            
            # Recent sets won ratio
            df[f'player{player_num}_sets_ratio'] = df.groupby(player_col)[sets_col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Games per set ratio (efficiency)
            df[f'player{player_num}_games_per_set'] = df.groupby(player_col).apply(
                lambda group: group[games_col].rolling(window=5, min_periods=1).sum() / 
                              (group[sets_col].rolling(window=5, min_periods=1).sum() + 0.1)
            ).reset_index(level=0, drop=True)
            
            # Momentum (last 3 vs previous matches)
            df[f'player{player_num}_momentum'] = (
                df.groupby(player_col)['result'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean() if player_num == 1
                    else (1 - x).rolling(window=3, min_periods=1).mean()
                ) - df[f'player{player_num}_form_10']
            )
        
        return df
    
    def _create_surface_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create surface-specific performance features"""
        df = df.copy()
        
        # Surface encoding
        surface_mapping = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
        df['surface_encoded'] = df['surface'].map(surface_mapping).fillna(0)
        
        for player_num in [1, 2]:
            player_col = f'player{player_num}'
            
            # Surface-specific win rate
            for surface in ['hard', 'clay', 'grass']:
                surface_matches = df[df['surface'] == surface]
                if not surface_matches.empty:
                    df[f'player{player_num}_{surface}_win_rate'] = df.groupby(player_col).apply(
                        lambda group: group[group['surface'] == surface]['result'].mean() 
                        if player_num == 1 else (1 - group[group['surface'] == surface]['result']).mean()
                    ).fillna(0.5).reindex(df[player_col]).values
                else:
                    df[f'player{player_num}_{surface}_win_rate'] = 0.5
            
            # Matches played on current surface
            df[f'player{player_num}_surface_experience'] = df.groupby([player_col, 'surface']).cumcount()
            
            # Surface adaptation (recent performance on surface)
            df[f'player{player_num}_surface_form'] = df.groupby([player_col, 'surface'])['result'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean() if player_num == 1
                else (1 - x).rolling(window=5, min_periods=1).mean()
            )
        
        # Surface advantage (difference in surface win rates)
        current_surface = df['surface'].iloc[0] if not df.empty else 'hard'
        if f'player1_{current_surface}_win_rate' in df.columns:
            df['surface_advantage'] = (df[f'player1_{current_surface}_win_rate'] - 
                                     df[f'player2_{current_surface}_win_rate'])
        
        return df
    
    def _create_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ranking-based features"""
        df = df.copy()
        
        # Basic ranking features
        df['ranking_difference'] = df['player1_rank'] - df['player2_rank']
        df['higher_ranked_player'] = (df['ranking_difference'] < 0).astype(int)
        
        # Ranking categories
        df['player1_top10'] = (df['player1_rank'] <= 10).astype(int)
        df['player2_top10'] = (df['player2_rank'] <= 10).astype(int)
        df['player1_top50'] = (df['player1_rank'] <= 50).astype(int)
        df['player2_top50'] = (df['player2_rank'] <= 50).astype(int)
        
        # Ranking upset potential
        df['upset_potential'] = np.maximum(0, df['ranking_difference'] / 100)  # Normalized
        
        # Historical ranking performance
        for player_num in [1, 2]:
            player_col = f'player{player_num}'
            rank_col = f'player{player_num}_rank'
            
            # Average ranking over time
            df[f'player{player_num}_avg_rank'] = df.groupby(player_col)[rank_col].transform(
                lambda x: x.expanding().mean()
            )
            
            # Ranking trend (improving/declining)
            df[f'player{player_num}_rank_trend'] = df.groupby(player_col)[rank_col].transform(
                lambda x: x.rolling(window=5, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0]  # Slope of ranking change
                )
            ).fillna(0)
        
        return df
    
    def train_tennis_model(self, df: pd.DataFrame) -> Dict:
        """Train enhanced tennis prediction model"""
        
        print(f"{Fore.CYAN}🔧 Training tennis model with surface analysis...{Style.RESET_ALL}")
        
        # Prepare features
        feature_df = self.prepare_tennis_features(df)
        
        # Select features for training
        exclude_cols = [
            'date', 'player1', 'player2', 'winner', 'tournament', 'round',
            'tour', 'year', 'match_id', 'result', 'surface'
        ]
        
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = feature_df[feature_cols].fillna(0)
        y = feature_df['result']  # Binary classification (0 or 1)
        
        # Train ensemble model
        model_result = self.model_trainer.train_ensemble_model(X, y, self.sport)
        
        self.model_data = model_result
        
        print(f"{Fore.GREEN}✅ Tennis model trained with accuracy: {model_result['best_score']:.3f}{Style.RESET_ALL}")
        
        return model_result
    
    def predict_match(self, player1: str, player2: str, surface: str = 'hard',
                     tournament: str = 'ATP 250') -> Dict[str, float]:
        """Predict tennis match outcome"""
        
        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model available. Please train first.{Style.RESET_ALL}")
            return {}
        
        # Create feature vector for prediction
        match_features = self._create_match_features(player1, player2, surface, tournament)
        
        if match_features is None:
            return {}
        
        # Get model prediction
        model = self.model_data['best_model']
        
        # Scale and transform features
        scaler = self.model_trainer.scalers.get('scaler')
        if scaler:
            match_features_scaled = scaler.transform([match_features])
        else:
            match_features_scaled = [match_features]
        
        feature_selector = self.model_trainer.feature_selectors.get('selector')
        if feature_selector:
            match_features_scaled = feature_selector.transform(match_features_scaled)
        
        # Get probabilities
        probabilities = model.predict_proba(match_features_scaled)[0]
        
        # Get ELO-based probabilities
        elo_probs = self.elo_system.get_match_probability(
            player1, player2, home_advantage=False
        )
        
        # Combine predictions (tennis models are typically very reliable)
        model_weight = 0.8
        elo_weight = 0.2
        
        final_probs = {
            'player1_win': model_weight * probabilities[1] + elo_weight * elo_probs[0],
            'player2_win': model_weight * probabilities[0] + elo_weight * elo_probs[2]
        }
        
        # Add set betting predictions
        if final_probs['player1_win'] > 0.6:
            final_probs['player1_straight_sets'] = 0.4
        elif final_probs['player1_win'] > 0.55:
            final_probs['player1_straight_sets'] = 0.25
        else:
            final_probs['player1_straight_sets'] = 0.15
        
        # Add total games prediction (simplified)
        avg_games = {'hard': 22, 'clay': 24, 'grass': 20}
        final_probs['predicted_total_games'] = avg_games.get(surface, 22)
        
        # Confidence and recommendation
        confidence = abs(final_probs['player1_win'] - final_probs['player2_win'])
        final_probs['confidence'] = confidence
        
        if confidence > CONFIG['ml'].min_confidence_threshold:
            best_outcome = 'player1_win' if final_probs['player1_win'] > final_probs['player2_win'] else 'player2_win'
            final_probs['recommendation'] = best_outcome
        else:
            final_probs['recommendation'] = 'no_bet'
        
        return final_probs
    
    def _create_match_features(self, player1: str, player2: str, surface: str,
                              tournament: str) -> Optional[List[float]]:
        """Create feature vector for match prediction"""
        
        try:
            player1_elo = self.elo_system.get_rating(player1)
            player2_elo = self.elo_system.get_rating(player2)
            
            # Surface encoding
            surface_mapping = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
            surface_encoded = surface_mapping.get(surface, 0)
            
            # Tournament importance
            is_grand_slam = any(gs in tournament.lower() for gs in self.grand_slams.keys())
            is_masters = any(m in tournament.lower() for m in self.masters_1000.keys())
            
            # Basic features for tennis
            features = [
                player1_elo,
                player2_elo,
                player1_elo - player2_elo,
                surface_encoded,
                int(is_grand_slam),
                int(is_masters),
                datetime.now().month,
                datetime.now().weekday(),
                0,  # Player 1 form (would be calculated from recent matches)
                0,  # Player 2 form
                0.5,  # Player 1 surface win rate
                0.5,  # Player 2 surface win rate
                100,  # Player 1 rank (would be fetched)
                100,  # Player 2 rank
                0,  # Ranking difference
                0,  # Surface experience
                0   # H2H record
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
            logger.error(f"Error creating match features: {e}")
            return None
    
    def get_upcoming_matches(self, tour: str = 'ATP') -> List[Dict]:
        """Get upcoming tennis matches"""
        
        cache_key = f"upcoming_tennis_{tour}"
        
        headers = {'X-RapidAPI-Key': self.config['api_key']}
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"{self.config['api_base']}/schedule/{today}/{tour.lower()}"
        
        api_data = self.data_manager.fetch_api_data(url, headers, cache_key=cache_key)
        
        upcoming_matches = []
        
        if api_data and 'matches' in api_data:
            for match in api_data['matches'][:10]:  # Limit to next 10 matches
                if match.get('status') == 'scheduled':
                    match_info = {
                        'date': match['start_date'][:10],
                        'time': match['start_date'][11:16],
                        'player1': match['competitors'][0]['name'],
                        'player2': match['competitors'][1]['name'],
                        'player1_rank': match['competitors'][0].get('rank', 999),
                        'player2_rank': match['competitors'][1].get('rank', 999),
                        'surface': match.get('surface', 'hard'),
                        'tournament': match.get('tournament_name', 'Unknown'),
                        'round': match.get('round', 'R1'),
                        'tour': tour,
                        'match_id': match['id']
                    }
                    upcoming_matches.append(match_info)
        
        return upcoming_matches
    
    def analyze_upcoming_matches(self, tour: str = 'ATP') -> List[Dict]:
        """Analyze upcoming matches and provide recommendations"""
        
        upcoming_matches = self.get_upcoming_matches(tour)
        recommendations = []
        
        print(f"{Fore.CYAN}🔍 Analyzing upcoming {self.tours.get(tour, tour)} matches...{Style.RESET_ALL}")
        
        for match in upcoming_matches:
            prediction = self.predict_match(
                match['player1'],
                match['player2'],
                match['surface'],
                match['tournament']
            )
            
            if prediction:
                recommendation = {
                    'match': f"{match['player1']} vs {match['player2']}",
                    'date': match['date'],
                    'time': match['time'],
                    'surface': match['surface'],
                    'tournament': match['tournament'],
                    'round': match['round'],
                    'player1_rank': match['player1_rank'],
                    'player2_rank': match['player2_rank'],
                    'predictions': prediction,
                    'confidence': prediction.get('confidence', 0),
                    'recommended_bet': prediction.get('recommendation', 'no_bet')
                }
                recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations
    
    def display_predictions(self, recommendations: List[Dict]):
        """Display tennis predictions"""
        
        if not recommendations:
            print(f"{Fore.YELLOW}⚠️ No betting opportunities found{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎾 TENNIS BETTING PREDICTIONS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'='*60}{Style.RESET_ALL}")
        
        for i, rec in enumerate(recommendations, 1):
            if rec['recommended_bet'] != 'no_bet':
                print(f"\n{Fore.CYAN}{Style.BRIGHT}{i}. {rec['match']}{Style.RESET_ALL}")
                print(f"   📅 {rec['date']} {rec['time']} | 🏆 {rec['tournament']}")
                print(f"   🎾 Surface: {rec['surface'].title()} | Round: {rec['round']}")
                print(f"   📊 Rankings: #{rec['player1_rank']} vs #{rec['player2_rank']}")
                print(f"   🎯 Recommendation: {Fore.GREEN}{rec['recommended_bet'].replace('_', ' ').title()}{Style.RESET_ALL}")
                print(f"   📈 Confidence: {Fore.YELLOW}{rec['confidence']:.1%}{Style.RESET_ALL}")
                
                preds = rec['predictions']
                player1_name = rec['match'].split(' vs ')[0]
                player2_name = rec['match'].split(' vs ')[1]
                
                print(f"   🎯 Probabilities:")
                print(f"      🥇 {player1_name}: {preds.get('player1_win', 0):.1%}")
                print(f"      🥈 {player2_name}: {preds.get('player2_win', 0):.1%}")
                
                if 'player1_straight_sets' in preds:
                    print(f"      ⚡ Straight Sets: {preds['player1_straight_sets']:.1%}")
                if 'predicted_total_games' in preds:
                    print(f"      🎯 Total Games O/U: {preds['predicted_total_games']:.0f}")
    
    def run(self):
        """Main run method for tennis predictor"""
        
        while True:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}🎾 TENNIS PREDICTOR{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
            print(f"1. 🎓 Train Model")
            print(f"2. 🔮 Make Predictions")
            print(f"3. 📊 Model Performance")
            print(f"4. 🏆 Surface Analysis")
            print(f"5. ⚙️ Settings")
            print(f"6. 🔙 Return to Main Menu")
            
            try:
                choice = input(f"\n{Fore.YELLOW}Select option: {Style.RESET_ALL}").strip()
                
                if choice == '1':
                    self.run_training_mode()
                elif choice == '2':
                    self.run_prediction_mode()
                elif choice == '3':
                    self.show_model_performance()
                elif choice == '4':
                    self.show_surface_analysis()
                elif choice == '5':
                    self.show_settings()
                elif choice == '6':
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please select 1-6.{Style.RESET_ALL}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def run_training_mode(self):
        """Run training mode for tennis"""
        
        print(f"{Fore.YELLOW}🎓 Tennis Training Mode{Style.RESET_ALL}")
        print(f"Available tours: {', '.join(self.tours.keys())}")
        
        # Collect data from multiple tours
        all_data = []
        
        # Start with ATP data
        current_year = datetime.now().year
        atp_data = self.fetch_tennis_data('ATP', str(current_year))
        if atp_data is not None and len(atp_data) > 0:
            all_data.append(atp_data)
            print(f"Added {len(atp_data)} ATP matches")
        
        # Add WTA data if available
        wta_data = self.fetch_tennis_data('WTA', str(current_year))
        if wta_data is not None and len(wta_data) > 0:
            all_data.append(wta_data)
            print(f"Added {len(wta_data)} WTA matches")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Training on {len(combined_data)} matches")
            
            # Train model
            self.train_tennis_model(combined_data)
        else:
            print(f"{Fore.RED}❌ No training data available{Style.RESET_ALL}")
    
    def run_prediction_mode(self):
        """Run prediction mode for tennis"""
        
        print(f"{Fore.YELLOW}🔮 Tennis Prediction Mode{Style.RESET_ALL}")
        
        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model found. Please run training first.{Style.RESET_ALL}")
            return
        
        while True:
            print(f"\n{Fore.CYAN}Select tour for predictions:{Style.RESET_ALL}")
            for i, (code, name) in enumerate(self.tours.items(), 1):
                print(f"{i}. {name} ({code})")
            print(f"{len(self.tours) + 1}. Return to tennis menu")
            
            try:
                choice = int(input(f"{Fore.YELLOW}Enter choice: {Style.RESET_ALL}"))
                
                if choice == len(self.tours) + 1:
                    break
                elif 1 <= choice <= len(self.tours):
                    tour_code = list(self.tours.keys())[choice - 1]
                    recommendations = self.analyze_upcoming_matches(tour_code)
                    self.display_predictions(recommendations)
                else:
                    print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")
                    
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
            except KeyboardInterrupt:
                break
    
    def show_model_performance(self):
        """Display model performance metrics"""
        
        if not self.model_data:
            print(f"{Fore.RED}❌ No trained model available{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}📊 TENNIS MODEL PERFORMANCE{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
        print(f"Best Model: {self.model_data['best_model_name']}")
        print(f"Accuracy: {self.model_data['best_score']:.1%}")
        
        print(f"\nAll Model Scores:")
        for model_name, score in self.model_data['model_scores'].items():
            print(f"  {model_name}: {score:.1%}")
    
    def show_surface_analysis(self):
        """Display surface-specific analysis"""
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🏆 SURFACE ANALYSIS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
        
        for surface, name in self.surfaces.items():
            print(f"\n{Fore.CYAN}{name}:{Style.RESET_ALL}")
            print(f"  Characteristics: {self._get_surface_characteristics(surface)}")
            print(f"  Typical Match Length: {self._get_typical_match_length(surface)}")
            print(f"  Favors: {self._get_surface_player_type(surface)}")
    
    def _get_surface_characteristics(self, surface: str) -> str:
        """Get surface characteristics"""
        characteristics = {
            'hard': 'Medium pace, consistent bounce, neutral surface',
            'clay': 'Slow pace, high bounce, favors topspin and endurance',
            'grass': 'Fast pace, low bounce, favors serve-and-volley',
            'carpet': 'Very fast, low bounce, indoor surface'
        }
        return characteristics.get(surface, 'Unknown characteristics')
    
    def _get_typical_match_length(self, surface: str) -> str:
        """Get typical match length by surface"""
        lengths = {
            'hard': '2-3 hours',
            'clay': '3-4 hours',
            'grass': '1.5-2.5 hours',
            'carpet': '1.5-2 hours'
        }
        return lengths.get(surface, 'Unknown')
    
    def _get_surface_player_type(self, surface: str) -> str:
        """Get player types favored by surface"""
        player_types = {
            'hard': 'All-court players, balanced game',
            'clay': 'Baseline grinders, defensive players',
            'grass': 'Big servers, net players',
            'carpet': 'Aggressive players, big hitters'
        }
        return player_types.get(surface, 'Unknown')
    
    def show_settings(self):
        """Display current tennis settings"""
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}⚙️ TENNIS SETTINGS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'─'*30}{Style.RESET_ALL}")
        print(f"Target Accuracy: {CONFIG['ml'].target_accuracy:.1%}")
        print(f"Confidence Threshold: {CONFIG['ml'].min_confidence_threshold:.1%}")
        print(f"ELO K-Factor: {CONFIG['elo'].tennis_k_factor}")
        print(f"Available Markets: {', '.join(CONFIG['betting'].tennis_markets)}")
        print(f"Supported Surfaces: {', '.join(self.surfaces.keys())}")

if __name__ == "__main__":
    predictor = TennisPredictor()
    predictor.run()
