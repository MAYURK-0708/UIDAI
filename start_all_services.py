#!/usr/bin/env python3
"""
Aadhaar Observatory - Unified Startup Script
Launches both the API server and Streamlit dashboard
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.absolute()

def check_data_availability():
    """Check if processed data is available"""
    data_dir = PROJECT_ROOT / 'data' / 'processed'
    required_files = ['merged_with_features.csv', 'enrolment_with_anomalies.csv']
    
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        print("\n⚠️  WARNING: Missing processed data files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease run: python run_analysis.py")
        print("This will generate the required processed data files.\n")
        return False
    
    return True

def start_api_server():
    """Start Flask API server"""
    print("\n📡 Starting API Server...")
    print("   API Server will run on: http://localhost:5000")
    
    # Start API server in a separate process
    api_process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / 'api_server.py')],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(2)  # Wait for server to start
    return api_process

def start_streamlit_dashboard():
    """Start Streamlit dashboard"""
    print("\n📊 Starting Streamlit Dashboard...")
    print("   Streamlit Dashboard will run on: http://localhost:8501")
    
    # Start Streamlit in a separate process
    streamlit_process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 
         str(PROJECT_ROOT / 'dashboard' / 'app.py'),
         '--server.port=8501'],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return streamlit_process

def main():
    """Main startup routine"""
    
    print("\n" + "="*80)
    print("🇮🇳  AADHAAR OBSERVATORY - ADVANCED DASHBOARD SUITE")
    print("="*80)
    print("\nUIDI Data Hackathon 2026")
    print("Advanced Analytics Platform for India's Aadhaar Ecosystem\n")
    
    # Check prerequisites
    print("📋 Checking prerequisites...")
    
    if not check_data_availability():
        print("✗ Data not available. Please run: python run_analysis.py")
        sys.exit(1)
    
    print("✓ Processed data found\n")
    
    # Start services
    processes = []
    
    try:
        # Start API server
        api_proc = start_api_server()
        processes.append(("API Server", api_proc))
        
        # Start Streamlit
        streamlit_proc = start_streamlit_dashboard()
        processes.append(("Streamlit Dashboard", streamlit_proc))
        
        print("\n" + "="*80)
        print("✅ SERVICES STARTED SUCCESSFULLY")
        print("="*80)
        
        print("\n🌐 Available Dashboards:")
        print("   1. Web Dashboard (HTML/JS):  http://localhost:5000 (open dashboard.html)")
        print("   2. Streamlit Dashboard:      http://localhost:8501")
        print("   3. API Server:               http://localhost:5000/api")
        
        print("\n📚 API Documentation:")
        print("   Health Check:     GET  /api/health")
        print("   Summary Stats:    GET  /api/summary")
        print("   States List:      GET  /api/states")
        print("   Districts:        GET  /api/districts")
        print("   Risk Assessment:  GET  /api/risk-assessment")
        print("   Regional Analysis: POST /api/regional-analysis")
        print("   Clustering:       POST /api/clustering")
        print("   Predictions:      POST /api/predictions")
        
        print("\n⌨️  Press Ctrl+C to stop all services\n")
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  {name} has stopped")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"✓ {name} stopped")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"✓ {name} force stopped")
        
        print("\n👋 Goodbye!\n")
        sys.exit(0)

if __name__ == '__main__':
    main()
