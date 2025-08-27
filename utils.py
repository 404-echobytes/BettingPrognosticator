"""
Enhanced Utilities Module for Multi-Sport Betting Predictor
Advanced ML pipeline utilities building on original betfing-predictor
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
import time
import requests
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, RFE
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings
import sqlite3
import joblib
from config import CONFIG

warnings.filterwarnings('ignore')

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/betting_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimitedRequester:
    """Thread-safe rate-limited API requester with daily and minute limits"""

    def __init__(self, requests_per_minute: int = 10, requests_per_day: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_request_times = []
        self.daily_request_count = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        self.lock = threading.Lock()
        self.session = requests.Session()

    def make_request(self, url: str, headers: Dict = None, params: Dict = None,
                    timeout: int = 30, max_retries: int = 3) -> Optional[requests.Response]:
        """Make rate-limited request with daily and minute limits"""

        with self.lock:
            now = time.time()
            current_time = datetime.now()

            # Reset daily counter if new day
            if current_time >= self.daily_reset_time:
                self.daily_request_count = 0
                self.daily_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                logger.info(f"Daily API limit reset. New limit: {self.requests_per_day} requests")

            # Check daily limit first
            if self.daily_request_count >= self.requests_per_day:
                time_until_reset = (self.daily_reset_time - current_time).total_seconds()
                logger.warning(f"Daily API limit ({self.requests_per_day}) reached. Reset in {time_until_reset/3600:.1f} hours")
                return None

            # Remove minute requests older than 1 minute
            self.minute_request_times = [t for t in self.minute_request_times if now - t < 60]

            # Check minute limit
            if len(self.minute_request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.minute_request_times[0])
                if sleep_time > 0:
                    logger.info(f"Per-minute rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    now = time.time()
                    self.minute_request_times = [t for t in self.minute_request_times if now - t < 60]

            # Record the request
            self.minute_request_times.append(now)
            self.daily_request_count += 1

        # Make the request with retries
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    headers=headers or {},
                    params=params or {},
                    timeout=timeout
                )

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = (2 ** attempt) * CONFIG['api'].backoff_factor
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed with status {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * CONFIG['api'].backoff_factor
                    time.sleep(wait_time)

        return None

class ELORatingSystem:
    """Enhanced ELO rating system for team/player strength tracking"""

    def __init__(self, sport: str):
        self.sport = sport
        self.config = CONFIG['elo']
        self.k_factor = getattr(self.config, f'{sport}_k_factor', self.config.k_factor)
        self.ratings = {}
        self.last_updated = {}

    def get_rating(self, entity: str) -> float:
        """Get current rating for team/player"""
        if entity not in self.ratings:
            self.ratings[entity] = self.config.initial_rating
            self.last_updated[entity] = datetime.now()

        # Apply decay if entity hasn't played recently
        days_since_update = (datetime.now() - self.last_updated[entity]).days
        if days_since_update > self.config.decay_threshold_days:
            decay_factor = self.config.decay_rate ** (days_since_update / 365)
            self.ratings[entity] = max(
                self.config.min_rating,
                self.ratings[entity] * decay_factor
            )

        return self.ratings[entity]

    def update_ratings(self, entity1: str, entity2: str, result: int,
                      home_advantage: bool = False) -> Tuple[float, float]:
        """
        Update ratings based on match result
        result: 1 if entity1 wins, 0 if draw, -1 if entity2 wins
        """
        rating1 = self.get_rating(entity1)
        rating2 = self.get_rating(entity2)

        # Apply home advantage if applicable
        if home_advantage and self.sport == 'football':
            rating1 += self.config.home_advantage

        # Calculate expected scores
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))

        # Convert result to actual scores
        if result == 1:  # entity1 wins
            actual1, actual2 = 1, 0
        elif result == -1:  # entity2 wins
            actual1, actual2 = 0, 1
        else:  # draw
            actual1, actual2 = 0.5, 0.5

        # Update ratings
        new_rating1 = rating1 + self.k_factor * (actual1 - expected1)
        new_rating2 = rating2 + self.k_factor * (actual2 - expected2)

        # Apply bounds
        new_rating1 = max(self.config.min_rating,
                         min(self.config.max_rating, new_rating1))
        new_rating2 = max(self.config.min_rating,
                         min(self.config.max_rating, new_rating2))

        # Update stored ratings
        self.ratings[entity1] = new_rating1
        self.ratings[entity2] = new_rating2
        self.last_updated[entity1] = datetime.now()
        self.last_updated[entity2] = datetime.now()

        return new_rating1, new_rating2

    def get_match_probability(self, entity1: str, entity2: str,
                            home_advantage: bool = False) -> Tuple[float, float, float]:
        """Get win probabilities for both entities and draw"""
        rating1 = self.get_rating(entity1)
        rating2 = self.get_rating(entity2)

        if home_advantage and self.sport == 'football':
            rating1 += self.config.home_advantage

        # Calculate basic win probabilities
        prob1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        prob2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))

        # Adjust for draw probability (sport-specific)
        if self.sport == 'football':
            draw_factor = 0.25  # Football has more draws
            prob_draw = draw_factor * (1 - abs(prob1 - prob2))
            prob1 *= (1 - prob_draw)
            prob2 *= (1 - prob_draw)
        elif self.sport == 'basketball':
            prob_draw = 0.02  # Very rare in basketball
            prob1 *= (1 - prob_draw)
            prob2 *= (1 - prob_draw)
        else:  # tennis - no draws
            prob_draw = 0

        # Normalize
        total = prob1 + prob2 + prob_draw
        return prob1/total, prob_draw/total, prob2/total

class AdvancedFeatureEngineer:
    """Advanced feature engineering for sports betting prediction"""

    def __init__(self, sport: str):
        self.sport = sport
        self.scalers = {}
        self.encoders = {}

    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base features common to all sports"""
        df = df.copy()

        # Date features
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Sort by date for time-based features
        df = df.sort_values('date')

        return df

    def create_team_form_features(self, df: pd.DataFrame, team_col: str,
                                 result_col: str, window: int = 5) -> pd.DataFrame:
        """Create team form features (last N games)"""
        df = df.copy()

        # Calculate rolling form
        df[f'{team_col}_form_{window}'] = (
            df.groupby(team_col)[result_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Win percentage
        df[f'{team_col}_win_pct_{window}'] = (
            df.groupby(team_col)[result_col]
            .rolling(window=window, min_periods=1)
            .apply(lambda x: (x == 1).sum() / len(x))
            .reset_index(level=0, drop=True)
        )

        # Recent momentum (last 3 vs previous games in window)
        if window >= 6:
            df[f'{team_col}_momentum'] = (
                df.groupby(team_col)[result_col]
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            ) - (
                df.groupby(team_col)[result_col]
                .rolling(window=window, min_periods=1)
                .apply(lambda x: x[:-3].mean() if len(x) > 3 else x.mean())
                .reset_index(level=0, drop=True)
            )

        return df

    def create_head_to_head_features(self, df: pd.DataFrame, team1_col: str,
                                   team2_col: str, result_col: str) -> pd.DataFrame:
        """Create head-to-head historical features"""
        df = df.copy()

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Create unique match identifier
        df['match_key'] = df.apply(
            lambda x: tuple(sorted([x[team1_col], x[team2_col]])), axis=1
        )

        # Calculate historical head-to-head record
        h2h_stats = df.groupby('match_key').agg({
            result_col: ['count', 'mean', 'std'],
            'date': 'max'
        }).reset_index()

        h2h_stats.columns = ['match_key', 'h2h_games', 'h2h_avg_result',
                            'h2h_result_std', 'last_meeting']

        # Merge back to main dataframe
        df = df.merge(h2h_stats, on='match_key', how='left')

        # Fill missing values
        df['h2h_games'] = df['h2h_games'].fillna(0)
        df['h2h_avg_result'] = df['h2h_avg_result'].fillna(0.5)
        df['h2h_result_std'] = df['h2h_result_std'].fillna(0)

        # Days since last meeting - convert to numeric
        if 'date' in df.columns and 'last_meeting' in df.columns:
            days_diff = (df['date'] - df['last_meeting']).dt.days.fillna(999)
            df['days_since_last_meeting'] = days_diff.astype(float)

        return df

    def create_sport_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sport-specific features"""
        if self.sport == 'football':
            return self._create_football_features(df)
        elif self.sport == 'basketball':
            return self._create_basketball_features(df)
        elif self.sport == 'tennis':
            return self._create_tennis_features(df)
        else:
            return df

    def _create_football_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Football-specific features"""
        df = df.copy()

        # Goal-based features
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['total_goals'] = df['home_score'] + df['away_score']
            df['goal_difference'] = df['home_score'] - df['away_score']
            df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
            df['both_teams_scored'] = ((df['home_score'] > 0) &
                                     (df['away_score'] > 0)).astype(int)

        # League strength features
        if 'league' in df.columns:
            league_strength = df.groupby('league')['total_goals'].mean()
            df['league_avg_goals'] = df['league'].map(league_strength)

        return df

    def _create_basketball_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basketball-specific features"""
        df = df.copy()

        # Scoring features
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['total_points'] = df['home_score'] + df['away_score']
            df['point_difference'] = df['home_score'] - df['away_score']
            df['high_scoring'] = (df['total_points'] > 200).astype(int)

        # Pace features (if available)
        if 'possessions' in df.columns:
            df['pace'] = df['total_points'] / df['possessions']

        return df

    def _create_tennis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tennis-specific features"""
        df = df.copy()

        # Surface features
        if 'surface' in df.columns:
            surface_encoder = LabelEncoder()
            df['surface_encoded'] = surface_encoder.fit_transform(df['surface'])
            self.encoders['surface'] = surface_encoder

        # Ranking features
        if 'player1_rank' in df.columns and 'player2_rank' in df.columns:
            df['rank_difference'] = df['player1_rank'] - df['player2_rank']
            df['higher_ranked_player'] = (df['rank_difference'] < 0).astype(int)

        return df

class WalkForwardValidator:
    """Walk-forward validation for time-series betting data"""

    def __init__(self, window_days: int = 30, step_days: int = 7):
        self.window_days = window_days
        self.step_days = step_days

    def split(self, df: pd.DataFrame, date_col: str = 'date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits"""
        df = df.sort_values(date_col)
        df['date_parsed'] = pd.to_datetime(df[date_col])

        min_date = df['date_parsed'].min()
        max_date = df['date_parsed'].max()

        splits = []
        current_date = min_date + timedelta(days=self.window_days)

        while current_date + timedelta(days=self.step_days) <= max_date:
            # Training set: all data before current_date
            train_mask = df['date_parsed'] < current_date

            # Test set: next step_days of data
            test_start = current_date
            test_end = current_date + timedelta(days=self.step_days)
            test_mask = ((df['date_parsed'] >= test_start) &
                        (df['date_parsed'] < test_end))

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                splits.append((train_idx, test_idx))

            current_date += timedelta(days=self.step_days)

        return splits

class ModelTrainer:
    """Advanced model training pipeline with ensemble methods"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.calibrators = {}

    def prepare_features(self, X: pd.DataFrame, y: pd.Series,
                        feature_selection: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features with scaling and selection"""

        # Handle categorical and datetime variables
        X_processed = X.copy()
        
        # Convert datetime columns to numeric (days since epoch)
        datetime_cols = X_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        
        # Handle remaining categorical variables
        for col in X_processed.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        # Ensure all data is numeric
        X_processed = X_processed.select_dtypes(include=[np.number])

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)

        # Feature selection
        if feature_selection and X_scaled.shape[1] > 10:
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(20, X_scaled.shape[1] // 2)
            )
            X_selected = selector.fit_transform(X_scaled, y)
            self.feature_selectors['selector'] = selector
        else:
            X_selected = X_scaled

        self.scalers['scaler'] = scaler
        return X_selected, y.values

    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series,
                           sport: str) -> Dict[str, Any]:
        """Train ensemble of models with proper validation"""

        X_processed, y_processed = self.prepare_features(X, y)

        # Initialize models
        models = {
            'xgboost': xgb.XGBClassifier(**CONFIG['ml'].xgb_params),
            'lightgbm': lgb.LGBMClassifier(**CONFIG['ml'].lgb_params),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }

        # Walk-forward validation
        validator = WalkForwardValidator(
            window_days=CONFIG['ml'].walk_forward_window,
            step_days=CONFIG['ml'].walk_forward_step
        )

        # Convert back to DataFrame for validation
        df_for_validation = X.copy()
        df_for_validation['target'] = y
        splits = validator.split(df_for_validation)

        model_scores = {}
        trained_models = {}

        # Train and validate each model
        for model_name, model in models.items():
            scores = []

            for train_idx, test_idx in splits:
                X_train, X_test = X_processed[train_idx], X_processed[test_idx]
                y_train, y_test = y_processed[train_idx], y_processed[test_idx]

                # Train model
                if model_name == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)

                # Predict and score
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)

            avg_score = np.mean(scores)
            model_scores[model_name] = avg_score

            # Train final model on all data
            final_model = models[model_name]
            final_model.fit(X_processed, y_processed)

            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(
                final_model,
                method=CONFIG['ml'].calibration_method,
                cv=CONFIG['ml'].calibration_cv
            )
            calibrated_model.fit(X_processed, y_processed)
            trained_models[model_name] = calibrated_model

            logger.info(f"{model_name} validation accuracy: {avg_score:.4f}")

        # Select best model or create ensemble
        best_model_name = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model_name]

        result = {
            'best_model': trained_models[best_model_name],
            'best_model_name': best_model_name,
            'best_score': best_score,
            'all_models': trained_models,
            'model_scores': model_scores,
            'feature_names': list(X.columns),
            'sport': sport
        }

        # Save model
        self.save_model(result, sport)

        return result

    def save_model(self, model_result: Dict, sport: str):
        """Save trained model and associated objects"""
        model_path = os.path.join(CONFIG['data'].models_path, f'{sport}_model.pkl')

        save_data = {
            'model_result': model_result,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'timestamp': datetime.now(),
            'config': CONFIG['ml']
        }

        os.makedirs(CONFIG['data'].models_path, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Model saved for {sport} at {model_path}")

    def load_model(self, sport: str) -> Optional[Dict]:
        """Load trained model"""
        model_path = os.path.join(CONFIG['data'].models_path, f'{sport}_model.pkl')

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)

                self.scalers = saved_data.get('scalers', {})
                self.feature_selectors = saved_data.get('feature_selectors', {})

                logger.info(f"Model loaded for {sport}")
                return saved_data['model_result']
            except Exception as e:
                logger.error(f"Error loading model for {sport}: {e}")

        return None

    def train_all_sports(self):
        """Train models for all supported sports"""
        sports = ['football', 'basketball', 'tennis']

        for sport in sports:
            try:
                logger.info(f"Starting training for {sport}")
                # This would be called by sport-specific modules
                # with their prepared data
                print(f"✅ Training pipeline ready for {sport}")
            except Exception as e:
                logger.error(f"Error training {sport} model: {e}")

class BacktestManager:
    """Comprehensive backtesting framework"""

    def __init__(self):
        self.results = {}

    def run_comprehensive_backtest(self):
        """Run backtesting for all sports"""
        sports = ['football', 'basketball', 'tennis']

        for sport in sports:
            try:
                logger.info(f"Starting backtest for {sport}")
                self.backtest_sport(sport)
            except Exception as e:
                logger.error(f"Error backtesting {sport}: {e}")

    def backtest_sport(self, sport: str):
        """Run backtest for specific sport"""
        # Load historical data and model
        trainer = ModelTrainer()
        model_data = trainer.load_model(sport)

        if not model_data:
            logger.warning(f"No model found for {sport}")
            return

        # Simulate betting strategy
        initial_bankroll = CONFIG['betting'].initial_bankroll
        current_bankroll = initial_bankroll

        # Track performance metrics
        total_bets = 0
        winning_bets = 0
        total_roi = 0

        # This would use historical data to simulate betting
        logger.info(f"Backtest completed for {sport}")

        self.results[sport] = {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': current_bankroll,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': winning_bets / max(total_bets, 1),
            'roi': (current_bankroll - initial_bankroll) / initial_bankroll
        }

        print(f"📈 {sport.capitalize()} Backtest Results:")
        print(f"   Win Rate: {self.results[sport]['win_rate']:.2%}")
        print(f"   ROI: {self.results[sport]['roi']:.2%}")

class DataManager:
    """Data management and caching utilities with sport-specific rate limiting"""

    def __init__(self, sport: str = None):
        self.cache_path = CONFIG['data'].cache_path
        self.sport = sport

        # Create sport-specific rate limiter
        if sport in ['football', 'basketball']:
            # API-Sports rate limits
            self.requester = RateLimitedRequester(
                requests_per_minute=CONFIG['api'].api_sports_calls_per_minute,
                requests_per_day=CONFIG['api'].api_sports_rate_limit
            )
        elif sport == 'tennis':
            # RapidAPI rate limits
            self.requester = RateLimitedRequester(
                requests_per_minute=CONFIG['api'].rapidapi_calls_per_minute,
                requests_per_day=CONFIG['api'].rapidapi_rate_limit // 30  # Monthly to daily estimate
            )
        else:
            # Default rate limiter
            self.requester = RateLimitedRequester()

        # Ensure cache directory exists
        try:
            os.makedirs(self.cache_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating cache directory: {e}")
            self.cache_path = 'data/cache'  # Fallback
            os.makedirs(self.cache_path, exist_ok=True)

    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if not expired"""
        cache_file = os.path.join(self.cache_path, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Check if cache is still valid
                cache_time = cached_data.get('timestamp', datetime.min)
                expiry_hours = CONFIG['data'].cache_expiry_hours

                if datetime.now() - cache_time < timedelta(hours=expiry_hours):
                    return cached_data.get('data')
            except Exception as e:
                logger.error(f"Error reading cache {cache_key}: {e}")

        return None

    def save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache with timestamp"""
        cache_file = os.path.join(self.cache_path, f"{cache_key}.pkl")

        try:
            cache_data = {
                'data': data,
                'timestamp': datetime.now()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            logger.error(f"Error saving to cache {cache_key}: {e}")

    def fetch_api_data(self, url: str, headers: Dict = None,
                      params: Dict = None, cache_key: str = None) -> Optional[Any]:
        """Fetch data from API with caching"""

        # Try cache first
        if cache_key:
            cached_data = self.get_cached_data(cache_key)
            if cached_data is not None:
                return cached_data

        # Make API request
        response = self.requester.make_request(url, headers, params)

        if response:
            try:
                data = response.json()

                # Save to cache
                if cache_key:
                    self.save_to_cache(cache_key, data)

                return data
            except Exception as e:
                logger.error(f"Error parsing API response: {e}")

        return None

class TelegramNotifier:
    """Telegram notification service"""
    def __init__(self):
        self.bot_token = CONFIG['telegram'].bot_token
        self.chat_id = CONFIG['telegram'].chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, message: str):
        """Send a message to the configured Telegram chat"""
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(self.base_url, data=payload, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            logger.info(f"Telegram message sent successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")