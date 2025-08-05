#!/usr/bin/env python3
"""
Setup script for Insurance Classification System
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    print("🔧 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    required_packages = [
        ("pandas", "pd"),
        ("numpy", "np"), 
        ("sklearn", "sklearn"),
        ("sentence_transformers", "SentenceTransformer"),
        ("torch", "torch")
    ]
    
    success = True
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            success = False
    
    return success

def create_data_directories():
    """Create necessary data directories"""
    print("📁 Creating data directories...")
    
    directories = ["data", "data/output", "data/exports"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")

def check_data_files():
    """Check if required data files exist"""
    print("📊 Checking data files...")
    
    required_files = [
        "data/input/insurance_taxonomy - insurance_taxonomy.csv",
        "data/input/ml_insurance_challenge.csv"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Main setup function"""
    print("🏢 Insurance Classification System Setup")
    print("="*50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    # Create directories
    create_data_directories()
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Some data files are missing!")
        print("Please ensure the following files exist:")
        print("  - data/input/insurance_taxonomy - insurance_taxonomy.csv")
        print("  - data/input/ml_insurance_challenge.csv")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies!")
        print("Try manually: python3 -m pip install -r requirements.txt")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages failed to import!")
        print("You may need to install additional dependencies.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the system: python3 main.py")
    print("2. Or try examples: python3 example_usage.py")
    print("3. Read the README.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 