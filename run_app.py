#!/usr/bin/env python3

import subprocess
import os
import time
import sys

def run_streamlit():
    """Run Streamlit app with auto-configuration"""
    
    # Set environment variables to skip initial setup
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_PORT'] = '5000'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Run streamlit with all required flags
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', '5000',
        '--server.address', '0.0.0.0',
        '--browser.gatherUsageStats', 'false',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    try:
        print("🚀 Starting Multi-Sport Betting Predictor...")
        print("📊 Running on http://0.0.0.0:5000")
        print("🎯 Ready for predictions!")
        
        # Run the command with empty stdin to skip email prompt
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Send empty line to skip email prompt
        process.stdin.write('\n')
        process.stdin.flush()
        
        # Wait for the process to start properly
        time.sleep(3)
        
        # Keep the process running
        process.wait()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return True
    
    return True

if __name__ == "__main__":
    run_streamlit()