Multi-Sport Betting Predictor by Raks
Overview
A comprehensive multi-sport betting prediction system covering Football, Basketball, and Tennis with unified Streamlit dashboard.

User Preferences
Wants separate code files for each sport (Football, Basketball, Tennis)
Main app with sport selector interface
Clean, neat starter page with bold multi-symbol design
No complex dashboard - simple sport selection interface
Prefers concise, direct responses
Project Architecture
app.py - Main Streamlit application with sport selector
football_predictor.py - Football betting prediction logic
basketball_predictor.py - Basketball betting prediction logic
tennis_predictor.py - Tennis betting prediction logic
Machine learning models with XGBoost and calibrated probability estimation
ELO rating system for team/player strength tracking
Rate-limited API integration for real sports data
Kelly Criterion-based bet sizing with bankroll management
Recent Changes
Created modular sport-specific prediction files
Built unified Streamlit interface with sport selection
Implemented consistent ML pipeline across all sports
Added professional styling with sport-specific branding
Technical Stack
Streamlit for web interface
XGBoost for ML predictions
scikit-learn for model calibration
pandas/numpy for data processing
requests for API integration
Rate limiting and caching for external APIs
External Dependencies
Sports data APIs for match/game information
Betting odds APIs for market data
Machine learning stack for predictions
Thread-safe operations for concurrent processing
