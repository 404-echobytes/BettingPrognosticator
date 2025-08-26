#!/usr/bin/env python3
"""
Enhanced Modular Betting Predictor
Supporting Football, Basketball, and Tennis with 70%+ Accuracy Target

Built upon the original betfing-predictor with advanced ML pipeline enhancements
"""

import os
import sys
import time
from datetime import datetime
from colorama import init, Fore, Back, Style
import argparse

# Initialize colorama for cross-platform colored output
init(autoreset=True)

def display_ascii_logo():
    """Display stylish ASCII art logo"""
    logo = f"""
{Fore.CYAN}{Style.BRIGHT}
███████╗███╗   ██╗██╗  ██╗ █████╗ ███╗   ██╗ ██████╗███████╗██████╗ 
██╔════╝████╗  ██║██║  ██║██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗
█████╗  ██╔██╗ ██║███████║███████║██╔██╗ ██║██║     █████╗  ██║  ██║
██╔══╝  ██║╚██╗██║██╔══██║██╔══██║██║╚██╗██║██║     ██╔══╝  ██║  ██║
███████╗██║ ╚████║██║  ██║██║  ██║██║ ╚████║╚██████╗███████╗██████╔╝
╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═════╝ 
                                                                      
██████╗ ███████╗████████╗████████╗██╗███╗   ██╗ ██████╗               
██╔══██╗██╔════╝╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝               
██████╔╝█████╗     ██║      ██║   ██║██╔██╗ ██║██║  ███╗              
██╔══██╗██╔══╝     ██║      ██║   ██║██║╚██╗██║██║   ██║              
██████╔╝███████╗   ██║      ██║   ██║██║ ╚████║╚██████╔╝              
╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝               
                                                                      
██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗ ██████╗ ██████╗  
██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗ 
██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║   ██║██████╔╝ 
██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║   ██║██╔══██╗ 
██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ╚██████╔╝██║  ██║ 
╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ 
{Style.RESET_ALL}"""
    
    print(logo)
    print(f"{Fore.YELLOW}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}🎯 Advanced ML Betting Predictor - Target Accuracy: 70%+{Style.RESET_ALL}")
    print(f"{Fore.WHITE}📊 Sports: Football ⚽ | Basketball 🏀 | Tennis 🎾{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}🤖 ML Stack: XGBoost | LightGBM | Ensemble Models{Style.RESET_ALL}")
    print(f"{Fore.CYAN}⚡ Features: ELO Ratings | Time-Series CV | Walk-Forward Validation{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print()

def display_loading_animation(duration=3):
    """Display loading animation"""
    loading_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    end_time = time.time() + duration
    
    while time.time() < end_time:
        for char in loading_chars:
            if time.time() >= end_time:
                break
            print(f"\r{Fore.CYAN}{Style.BRIGHT}🔄 Initializing Enhanced ML Pipeline {char}{Style.RESET_ALL}", end="", flush=True)
            time.sleep(0.1)
    
    print(f"\r{Fore.GREEN}{Style.BRIGHT}✅ ML Pipeline Ready!{Style.RESET_ALL}" + " " * 20)
    print()

def display_sport_menu():
    """Display sport selection menu"""
    print(f"{Fore.YELLOW}{Style.BRIGHT}🏆 SELECT SPORT FOR PREDICTION{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'─' * 50}{Style.RESET_ALL}")
    print()
    
    sports = [
        ("⚽", "Football", "Enhanced from original with advanced features"),
        ("🏀", "Basketball", "NBA/League predictions with player stats"),
        ("🎾", "Tennis", "ATP/WTA with surface analysis"),
        ("📊", "Dashboard", "Web interface with real-time monitoring"),
        ("⚙️", "Model Training", "Retrain models with latest data"),
        ("📈", "Backtest", "Validate model performance"),
        ("❌", "Exit", "Close application")
    ]
    
    for i, (emoji, name, desc) in enumerate(sports, 1):
        print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {emoji} {Fore.CYAN}{Style.BRIGHT}{name:<12}{Style.RESET_ALL} - {Fore.WHITE}{desc}{Style.RESET_ALL}")
    
    print()
    print(f"{Fore.WHITE}{'─' * 50}{Style.RESET_ALL}")

def get_user_choice():
    """Get and validate user choice"""
    while True:
        try:
            choice = input(f"{Fore.YELLOW}📋 Enter your choice (1-7): {Style.RESET_ALL}").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7']:
                return int(choice)
            else:
                print(f"{Fore.RED}❌ Invalid choice. Please enter 1-7.{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
            sys.exit(0)
        except Exception:
            print(f"{Fore.RED}❌ Invalid input. Please enter a number 1-7.{Style.RESET_ALL}")

def display_system_info():
    """Display system information and configuration"""
    print(f"{Fore.CYAN}{Style.BRIGHT}🔧 SYSTEM CONFIGURATION{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'─' * 30}{Style.RESET_ALL}")
    
    # Check data directories
    data_dirs = ['data/models', 'data/cache', 'data/history']
    for dir_path in data_dirs:
        status = "✅" if os.path.exists(dir_path) else "❌"
        print(f"{status} {dir_path}")
    
    # Check environment variables
    api_keys = {
        'FOOTBALL_API_KEY': os.getenv('FOOTBALL_API_KEY', 'Not Set'),
        'BASKETBALL_API_KEY': os.getenv('BASKETBALL_API_KEY', 'Not Set'),
        'TENNIS_API_KEY': os.getenv('TENNIS_API_KEY', 'Not Set')
    }
    
    print(f"\n{Fore.YELLOW}🔑 API Configuration:{Style.RESET_ALL}")
    for key, value in api_keys.items():
        status = "✅" if value != 'Not Set' else "⚠️"
        masked_value = value[:8] + "..." if value != 'Not Set' else "Not Set"
        print(f"{status} {key}: {masked_value}")
    
    print()

def run_football_predictor():
    """Launch football prediction module"""
    print(f"{Fore.GREEN}{Style.BRIGHT}⚽ Loading Football Predictor...{Style.RESET_ALL}")
    try:
        from football import FootballPredictor
        predictor = FootballPredictor()
        predictor.run()
    except ImportError as e:
        print(f"{Fore.RED}❌ Error importing football module: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error running football predictor: {e}{Style.RESET_ALL}")

def run_basketball_predictor():
    """Launch basketball prediction module"""
    print(f"{Fore.GREEN}{Style.BRIGHT}🏀 Loading Basketball Predictor...{Style.RESET_ALL}")
    try:
        from basketball import BasketballPredictor
        predictor = BasketballPredictor()
        predictor.run()
    except ImportError as e:
        print(f"{Fore.RED}❌ Error importing basketball module: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error running basketball predictor: {e}{Style.RESET_ALL}")

def run_tennis_predictor():
    """Launch tennis prediction module"""
    print(f"{Fore.GREEN}{Style.BRIGHT}🎾 Loading Tennis Predictor...{Style.RESET_ALL}")
    try:
        from tennis import TennisPredictor
        predictor = TennisPredictor()
        predictor.run()
    except ImportError as e:
        print(f"{Fore.RED}❌ Error importing tennis module: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error running tennis predictor: {e}{Style.RESET_ALL}")

def run_web_dashboard():
    """Launch web dashboard"""
    print(f"{Fore.GREEN}{Style.BRIGHT}📊 Starting Web Dashboard...{Style.RESET_ALL}")
    try:
        import subprocess
        import sys
        print(f"{Fore.CYAN}🌐 Dashboard will be available at: http://0.0.0.0:5000{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📱 Telegram notifications enabled{Style.RESET_ALL}")
        
        # Run Streamlit app
        subprocess.run([sys.executable, 'run_app.py'])
        
    except ImportError as e:
        print(f"{Fore.RED}❌ Error importing required modules: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error running web dashboard: {e}{Style.RESET_ALL}")

def run_model_training():
    """Launch model training pipeline"""
    print(f"{Fore.GREEN}{Style.BRIGHT}⚙️ Starting Model Training Pipeline...{Style.RESET_ALL}")
    try:
        from utils import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_all_sports()
    except ImportError as e:
        print(f"{Fore.RED}❌ Error importing training module: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error running model training: {e}{Style.RESET_ALL}")

def run_backtesting():
    """Launch backtesting framework"""
    print(f"{Fore.GREEN}{Style.BRIGHT}📈 Starting Backtesting Framework...{Style.RESET_ALL}")
    try:
        from utils import BacktestManager
        backtester = BacktestManager()
        backtester.run_comprehensive_backtest()
    except ImportError as e:
        print(f"{Fore.RED}❌ Error importing backtest module: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error running backtesting: {e}{Style.RESET_ALL}")

def create_data_directories():
    """Create necessary data directories"""
    directories = [
        'data',
        'data/models',
        'data/cache', 
        'data/history',
        'web/static',
        'web/templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main application entry point"""
    # Create necessary directories
    create_data_directories()
    
    # Clear screen (cross-platform)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display startup sequence
    display_ascii_logo()
    display_loading_animation()
    display_system_info()
    
    # Main application loop
    while True:
        display_sport_menu()
        choice = get_user_choice()
        
        print()  # Add spacing
        
        if choice == 1:
            run_football_predictor()
        elif choice == 2:
            run_basketball_predictor()
        elif choice == 3:
            run_tennis_predictor()
        elif choice == 4:
            run_web_dashboard()
        elif choice == 5:
            run_model_training()
        elif choice == 6:
            run_backtesting()
        elif choice == 7:
            print(f"{Fore.YELLOW}{Style.BRIGHT}👋 Thank you for using Enhanced Betting Predictor!{Style.RESET_ALL}")
            print(f"{Fore.GREEN}🎯 Keep targeting that 70%+ accuracy!{Style.RESET_ALL}")
            sys.exit(0)
        
        # Return to menu after operation
        input(f"\n{Fore.CYAN}Press Enter to return to main menu...{Style.RESET_ALL}")
        os.system('cls' if os.name == 'nt' else 'clear')
        display_ascii_logo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Application terminated by user.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        sys.exit(1)
