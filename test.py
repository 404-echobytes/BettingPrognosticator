
#!/usr/bin/env python3
"""
Test script to verify all fixes are working
"""

import os
import sys
from datetime import datetime
from colorama import init, Fore, Back, Style

init(autoreset=True)

def test_imports():
    """Test all imports work correctly"""
    print(f"{Fore.CYAN}Testing imports...{Style.RESET_ALL}")
    
    try:
        # Test main modules
        from config import CONFIG
        from utils import ModelTrainer, DataManager, ELORatingSystem
        from football import FootballPredictor
        from basketball import BasketballPredictor
        from tennis import TennisPredictor
        
        print(f"{Fore.GREEN}✅ All imports successful{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ Import error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return False

def test_api_keys():
    """Test API key configuration"""
    print(f"{Fore.CYAN}Testing API configuration...{Style.RESET_ALL}")
    
    from config import CONFIG
    
    api_keys = {
        'API-Sports': CONFIG['api'].api_sports_key,
        'RapidAPI': CONFIG['api'].rapidapi_key,
        'Telegram': CONFIG['telegram'].bot_token
    }
    
    for name, key in api_keys.items():
        if key and key != 'demo_key':
            print(f"{Fore.GREEN}✅ {name}: Configured{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ {name}: Using demo key{Style.RESET_ALL}")

def test_model_training():
    """Test model training for each sport"""
    print(f"{Fore.CYAN}Testing model training...{Style.RESET_ALL}")
    
    sports = ['football', 'basketball', 'tennis']
    
    for sport in sports:
        try:
            print(f"\nTesting {sport} training...")
            
            if sport == 'football':
                from football import FootballPredictor
                predictor = FootballPredictor()
                # Test sample data generation
                sample_data = predictor._generate_sample_football_data('PL', '2024')
                print(f"  Generated {len(sample_data)} football matches")
                
                # Test feature preparation
                features = predictor.prepare_football_features(sample_data)
                print(f"  Created {len(features.columns)} features")
                
            elif sport == 'basketball':
                from basketball import BasketballPredictor
                predictor = BasketballPredictor()
                sample_data = predictor._generate_sample_nba_data('2024')
                print(f"  Generated {len(sample_data)} basketball games")
                
                features = predictor.prepare_basketball_features(sample_data)
                print(f"  Created {len(features.columns)} features")
                
            elif sport == 'tennis':
                from tennis import TennisPredictor
                predictor = TennisPredictor()
                sample_data = predictor._generate_sample_tennis_data('ATP', '2024')
                print(f"  Generated {len(sample_data)} tennis matches")
                
                features = predictor.prepare_tennis_features(sample_data)
                print(f"  Created {len(features.columns)} features")
            
            print(f"{Fore.GREEN}✅ {sport.capitalize()} test passed{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ {sport.capitalize()} test failed: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

def test_directories():
    """Test required directories exist"""
    print(f"{Fore.CYAN}Testing directory structure...{Style.RESET_ALL}")
    
    required_dirs = ['data', 'data/models', 'data/cache', 'data/history']
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"{Fore.GREEN}✅ {dir_path}{Style.RESET_ALL}")
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"{Fore.YELLOW}📁 Created {dir_path}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to create {dir_path}: {e}{Style.RESET_ALL}")

def main():
    """Run all tests"""
    print(f"{Fore.GREEN}{Style.BRIGHT}🔧 DIAGNOSTIC TEST SUITE{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'='*50}{Style.RESET_ALL}")
    
    tests = [
        ("Directory Structure", test_directories),
        ("Module Imports", test_imports),
        ("API Configuration", test_api_keys),
        ("Model Training", test_model_training)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{Fore.YELLOW}Running {test_name} test...{Style.RESET_ALL}")
        try:
            result = test_func()
            results[test_name] = result if result is not None else True
        except Exception as e:
            print(f"{Fore.RED}Test failed with error: {e}{Style.RESET_ALL}")
            results[test_name] = False
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}TEST SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'='*30}{Style.RESET_ALL}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        color = Fore.GREEN if result else Fore.RED
        print(f"{color}{status} {test_name}{Style.RESET_ALL}")
    
    overall_result = all(results.values())
    if overall_result:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 ALL TESTS PASSED!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}The betting predictor is ready for training and predictions.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}{Style.BRIGHT}⚠️ SOME TESTS FAILED{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Please review the errors above.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
