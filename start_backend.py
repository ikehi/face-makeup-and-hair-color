#!/usr/bin/env python3
"""
Startup script for the Flask backend with your ACTUAL working models
"""

import os
import sys
import subprocess

def main():
    print("🚀 Starting Beauty AI Platform with your ACTUAL working models...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: Please run this script from the 'face-makeup and hair color' directory")
        print("   Current directory:", os.getcwd())
        return
    
    # Check if your actual model exists
    model_path = 'cp/79999_iter.pth'
    if not os.path.exists(model_path):
        print("❌ Error: Your actual working model not found!")
        print("   Expected path:", model_path)
        print("   Please ensure the model file exists")
        return
    
    print("✅ Found your actual working model:", model_path)
    print("✅ Model size:", f"{os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Install Python dependencies if needed
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install dependencies:")
        print(e.stderr.decode())
        return
    
    # Start the Flask backend
    print("\n🔥 Starting Flask backend with your actual working models...")
    print("   Backend will be available at: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting backend: {e}")

if __name__ == '__main__':
    main() 