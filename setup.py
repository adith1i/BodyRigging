#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Body Rigging Project Setup')
    parser.add_argument('--install', action='store_true',
                        help='Install required dependencies')
    parser.add_argument('--create-dirs', action='store_true',
                        help='Create necessary directories')
    parser.add_argument('--check', action='store_true',
                        help='Check if dependencies are installed')
    parser.add_argument('--all', action='store_true',
                        help='Perform all setup tasks')
    return parser.parse_args()

def install_dependencies():
    """Install required dependencies from requirements.txt"""
    print("Installing dependencies...")

    try:
        # First try with pip
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies using pip.")

        # Try with pip3 if pip fails
        try:
            subprocess.run(["pip3", "install", "-r", "requirements.txt"], check=True)
            print("Dependencies installed successfully using pip3!")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Failed to install dependencies using pip3.")

            print("\nPlease install dependencies manually by running:")
            print("pip install -r requirements.txt")
            return False

def create_directories():
    """Create necessary directories for the project"""
    print("Creating project directories...")

    directories = [
        "recordings",  # For recorded motion data
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    dependencies = [
        "cv2",    # OpenCV
        "mediapipe",
        "numpy",
        "matplotlib",
        "pygame"
    ]

    all_installed = True

    for dep in dependencies:
        try:
            if dep == "cv2":
                # Special case for OpenCV
                import cv2
                print(f"✓ {dep} (OpenCV) installed - version: {cv2.__version__}")
            else:
                # Import the module and check version if available
                module = __import__(dep)
                version = getattr(module, "__version__", "unknown")
                print(f"✓ {dep} installed - version: {version}")
        except ImportError:
            print(f"✗ {dep} not installed")
            all_installed = False

    return all_installed

def main():
    """Main setup function"""
    args = parse_args()

    # If no arguments are provided, display help
    if not any(vars(args).values()):
        print("No arguments specified. Use --help for usage information.")
        return 1

    # Track overall success
    success = True

    # Install dependencies
    if args.install or args.all:
        if not install_dependencies():
            success = False

    # Create directories
    if args.create_dirs or args.all:
        if not create_directories():
            success = False

    # Check dependencies
    if args.check or args.all:
        if not check_dependencies():
            success = False
            print("\nSome dependencies are missing. Please install them using:")
            print("pip install -r requirements.txt")

    # Final message
    if success:
        print("\nSetup completed successfully!")
        print("\nTo run the application with a webcam:")
        print("python main.py")
        print("\nTo run with additional features:")
        print("python main.py --flip --ui --analyzer")
        print("\nTo process a video file:")
        print("python run_demo.py --video path/to/video.mp4 --output output.mp4")
    else:
        print("\nSetup completed with some issues. Please address them before running the application.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
