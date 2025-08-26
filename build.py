
#!/usr/bin/env python3
"""
Enhanced Betting Predictor Build Script
Automatically sets up project structure and dependencies
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'data',
        'data/models',
        'data/cache', 
        'data/history',
        'web',
        'web/static',
        'web/templates',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_requirements_txt():
    """Create requirements.txt with all dependencies"""
    requirements = """
# Core ML and Data Science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0

# Web Framework
flask>=3.0.0
streamlit>=1.28.0

# Data Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# API and HTTP
requests>=2.31.0
aiohttp>=3.8.0

# Database
sqlite3

# Utilities
colorama>=0.4.6
python-dotenv>=1.0.0
joblib>=1.3.0

# Optional: Advanced features
scipy>=1.11.0
statsmodels>=0.14.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    print("✅ Created requirements.txt")

def setup_environment():
    """Set up environment configuration"""
    if not os.path.exists('.env'):
        shutil.copy('.env.example', '.env')
        print("✅ Created .env from template")
    else:
        print("⚠️  .env already exists, skipping...")

def install_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Data and Models
data/models/*.pkl
data/models/*.joblib
data/cache/*
!data/cache/.gitkeep
data/history/*
!data/history/.gitkeep
logs/*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Streamlit
.streamlit/secrets.toml
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ Created .gitignore")

def create_readme():
    """Create comprehensive README"""
    readme_content = """
# Enhanced Betting Predictor

Advanced ML-powered sports betting predictor supporting Football, Basketball, and Tennis with 70%+ accuracy target.

## Features

- 🎯 **Multi-Sport Support**: Football, Basketball, Tennis
- 🤖 **Advanced ML Pipeline**: XGBoost, LightGBM, Ensemble Models
- 📊 **ELO Rating System**: Dynamic team/player strength calculation
- 💰 **Kelly Criterion**: Optimal bet sizing
- 📱 **Telegram Integration**: Real-time notifications
- 🌐 **Web Dashboard**: Real-time monitoring interface
- 📈 **Backtesting Framework**: Historical performance validation

## Quick Start

1. **Setup Environment**:
   ```bash
   python build.py
   ```

2. **Configure API Keys**:
   Edit `.env` file with your API keys

3. **Run Application**:
   ```bash
   python main.py  # CLI interface
   python run_app.py  # Web interface
   ```

## API Keys Required

- **API-Sports**: Football & Basketball data (100 calls/day free)
- **RapidAPI**: Tennis data (free tier)
- **Odds API**: Betting odds data
- **Telegram Bot**: Notifications (optional)

## Project Structure

```
├── main.py              # Main CLI application
├── run_app.py           # Streamlit web interface
├── app.py               # Main web application
├── config.py            # Configuration management
├── utils.py             # Utility functions and ML pipeline
├── football.py          # Football predictor
├── basketball.py        # Basketball predictor
├── tennis.py            # Tennis predictor
├── web/
│   ├── app.py          # Flask web dashboard
│   ├── templates/      # HTML templates
│   └── static/         # CSS/JS assets
└── data/               # Data storage
    ├── models/         # Trained models
    ├── cache/          # API cache
    └── history/        # Prediction history
```

## Configuration

Edit `config.py` or `.env` to customize:
- ML model parameters
- Betting strategy settings
- API rate limiting
- Risk management rules

## Usage Examples

### CLI Interface
```bash
python main.py
# Select sport and follow prompts
```

### Web Dashboard
```bash
python run_app.py
# Open http://localhost:5000
```

### Telegram Notifications
Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`

## Model Performance

Target accuracy: **70%+**
- Football: Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- Basketball: NBA, EuroLeague
- Tennis: ATP, WTA Tours

## Risk Management

- Kelly Criterion for bet sizing
- Maximum 2% bankroll per bet
- Stop-loss protection
- Expected value thresholds

## Disclaimer

This software is for educational purposes only. Sports betting involves risk. Always gamble responsibly and within your means.

## License

MIT License - see LICENSE file for details.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content.strip())
    print("✅ Created README.md")

def main():
    """Main build function"""
    print("🚀 Enhanced Betting Predictor Build Script")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_requirements_txt()
    setup_environment()
    create_gitignore()
    create_readme()
    
    # Install dependencies
    if install_dependencies():
        print("\n✅ Build completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python main.py (CLI) or python run_app.py (Web)")
        print("3. Check README.md for detailed instructions")
    else:
        print("\n❌ Build failed during dependency installation")
        print("Please check your Python environment and try again")

if __name__ == "__main__":
    main()
