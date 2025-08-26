"""
Enhanced Configuration System for Multi-Sport Betting Predictor
Extended from original betfing-predictor configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

@dataclass
class APIConfig:
    """API configuration for various sports data sources"""
    # API-Sports (Football & Basketball) - 100 calls per day
    api_sports_key: str = field(default_factory=lambda: os.getenv('APISPORTS_API_KEY', 'demo_key'))
    api_sports_base: str = 'https://v3.football.api-sports.io'
    api_sports_basketball_base: str = 'https://v1.basketball.api-sports.io'
    api_sports_rate_limit: int = 100  # calls per day for free version
    api_sports_calls_per_minute: int = 10  # Rate limit per minute

    # RapidAPI for Tennis (Free version)
    rapidapi_key: str = field(default_factory=lambda: os.getenv('RAPIDAPI_KEY', 'demo_key'))
    rapidapi_tennis_base: str = 'https://ultimate-tennis1.p.rapidapi.com'
    rapidapi_rate_limit: int = 500  # calls per month for free
    rapidapi_calls_per_minute: int = 5  # Conservative rate limiting

    # Odds APIs (optional)
    odds_api_key: str = field(default_factory=lambda: os.getenv('ODDS_API_KEY', 'demo_key'))
    odds_api_base: str = 'https://api.the-odds-api.com/v4'

    # Request settings
    request_timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 1.0  # Longer backoff for limited APIs

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    # Model parameters
    target_accuracy: float = 0.70  # 70% target accuracy
    min_confidence_threshold: float = 0.65
    ensemble_models: List[str] = field(default_factory=lambda: ['xgboost', 'lightgbm', 'random_forest'])

    # XGBoost parameters (enhanced from original)
    xgb_params: Dict = field(default_factory=lambda: {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
        'eval_metric': 'mlogloss'
    })

    # LightGBM parameters
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    })

    # Cross-validation parameters
    cv_folds: int = 5
    cv_scoring: str = 'neg_log_loss'
    test_size: float = 0.2
    validation_strategy: str = 'time_series'  # 'time_series' or 'stratified'

    # Walk-forward validation
    walk_forward_window: int = 30  # days
    walk_forward_step: int = 7     # days

    # Feature engineering
    feature_selection_method: str = 'recursive'  # 'recursive', 'mutual_info', 'chi2'
    max_features: Optional[int] = None

    # Model calibration
    calibration_method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    calibration_cv: int = 3

@dataclass
class BettingConfig:
    """Betting strategy configuration"""
    # Bankroll management
    initial_bankroll: float = 1000.0
    max_bet_percentage: float = 0.02  # 2% max bet size
    min_bet_amount: float = 5.0
    max_bet_amount: float = 50.0

    # Kelly Criterion settings
    kelly_multiplier: float = 0.25  # Conservative Kelly
    use_fractional_kelly: bool = True

    # Risk management
    max_daily_risk: float = 0.10  # 10% of bankroll per day
    max_weekly_risk: float = 0.25  # 25% of bankroll per week
    stop_loss_percentage: float = 0.20  # 20% stop loss

    # Betting thresholds
    min_odds: float = 1.5
    max_odds: float = 5.0
    min_expected_value: float = 0.05  # 5% minimum EV

    # Market types by sport
    football_markets: List[str] = field(default_factory=lambda: [
        'match_winner', 'over_under_2.5', 'both_teams_score', 'double_chance'
    ])
    basketball_markets: List[str] = field(default_factory=lambda: [
        'match_winner', 'point_spread', 'over_under_total', 'first_half_winner'
    ])
    tennis_markets: List[str] = field(default_factory=lambda: [
        'match_winner', 'set_handicap', 'over_under_games', 'first_set_winner'
    ])

@dataclass
class ELOConfig:
    """ELO rating system configuration"""
    # Base ELO parameters
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    home_advantage: float = 100.0

    # Sport-specific adjustments
    football_k_factor: float = 20.0
    basketball_k_factor: float = 32.0
    tennis_k_factor: float = 24.0

    # Rating decay (for handling inactivity)
    decay_rate: float = 0.95
    decay_threshold_days: int = 365

    # Minimum games for stable rating
    min_games_for_rating: int = 10

    # Rating bounds
    min_rating: float = 800.0
    max_rating: float = 2800.0

@dataclass
class DataConfig:
    """Data management configuration"""
    # Data paths
    base_data_path: str = 'data'
    models_path: str = 'data/models'
    cache_path: str = 'data/cache'
    history_path: str = 'data/history'

    # Cache settings
    cache_expiry_hours: int = 24
    max_cache_size_mb: int = 500

    # Historical data
    min_historical_days: int = 365  # 1 year minimum
    max_historical_days: int = 1095  # 3 years maximum

    # Data validation
    required_columns: Dict[str, List[str]] = field(default_factory=lambda: {
        'football': ['date', 'home_team', 'away_team', 'home_score', 'away_score'],
        'basketball': ['date', 'home_team', 'away_team', 'home_score', 'away_score'],
        'tennis': ['date', 'player1', 'player2', 'winner', 'surface']
    })

@dataclass
class SystemConfig:
    """System configuration settings"""
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file: str = 'logs/betting_predictor.log'
    max_workers: int = 4
    timeout_seconds: int = 30
    retry_attempts: int = 3
    performance_tracking: bool = True

    # Web Configuration
    web_host: str = '0.0.0.0'
    web_port: int = 5000
    web_debug: bool = False

@dataclass
class TelegramConfig:
    """Telegram Bot Configuration"""
    bot_token: str = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN', '8427390358:AAFtZ34EGUlFWF2DfLopXYJM9ME5tm0WMsc'))
    chat_id: str = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID', '6123696396'))
    enabled: bool = True


# Global configuration instance
CONFIG = {
    'api': APIConfig(),
    'ml': MLConfig(),
    'elo': ELOConfig(),
    'betting': BettingConfig(),
    'data': DataConfig(),
    'system': SystemConfig(),
    'telegram': TelegramConfig()
}

def load_config_from_file(config_path: str = 'config.json') -> Dict:
    """Load configuration from JSON file if it exists"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)

            # Update CONFIG with file values
            for section, values in file_config.items():
                if section in CONFIG:
                    for key, value in values.items():
                        if hasattr(CONFIG[section], key):
                            setattr(CONFIG[section], key, value)

            print(f"✅ Configuration loaded from {config_path}")
        except Exception as e:
            print(f"⚠️ Error loading config file: {e}")

    return CONFIG

def save_config_to_file(config_path: str = 'config.json') -> bool:
    """Save current configuration to JSON file"""
    try:
        config_dict = {}
        for section_name, section_config in CONFIG.items():
            config_dict[section_name] = {}
            for field_name in section_config.__dataclass_fields__:
                config_dict[section_name][field_name] = getattr(section_config, field_name)

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        print(f"✅ Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving config file: {e}")
        return False

def get_sport_config(sport: str) -> Dict:
    """Get sport-specific configuration"""
    if sport == 'football':
        return {
            'api_key': CONFIG['api'].api_sports_key,
            'api_base': CONFIG['api'].api_sports_base,
            'rate_limit_daily': CONFIG['api'].api_sports_rate_limit,
            'rate_limit_minute': CONFIG['api'].api_sports_calls_per_minute,
            'k_factor': CONFIG['elo'].football_k_factor,
            'markets': CONFIG['betting'].football_markets,
            'required_columns': CONFIG['data'].required_columns['football']
        }
    elif sport == 'basketball':
        return {
            'api_key': CONFIG['api'].api_sports_key,
            'api_base': CONFIG['api'].api_sports_basketball_base,
            'rate_limit_daily': CONFIG['api'].api_sports_rate_limit,
            'rate_limit_minute': CONFIG['api'].api_sports_calls_per_minute,
            'k_factor': CONFIG['elo'].basketball_k_factor,
            'markets': CONFIG['betting'].basketball_markets,
            'required_columns': CONFIG['data'].required_columns['basketball']
        }
    elif sport == 'tennis':
        return {
            'api_key': CONFIG['api'].rapidapi_key,
            'api_base': CONFIG['api'].rapidapi_tennis_base,
            'rate_limit_daily': CONFIG['api'].rapidapi_rate_limit,
            'rate_limit_minute': CONFIG['api'].rapidapi_calls_per_minute,
            'k_factor': CONFIG['elo'].tennis_k_factor,
            'markets': CONFIG['betting'].tennis_markets,
            'required_columns': CONFIG['data'].required_columns['tennis']
        }

    return {}

# Initialize configuration on import
CONFIG = load_config_from_file()