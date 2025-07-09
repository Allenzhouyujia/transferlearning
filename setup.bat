@echo off
chcp 65001 >nul
title CIFAR-10 Deep Learning Course - Environment Setup Wizard

echo.
echo ================================================================
echo 🎓 CIFAR-10 Deep Learning Course - Environment Setup Wizard
echo ================================================================
echo.
echo 🚀 Launching automated environment configuration...
echo 🎯 This script will magically configure your Python virtual environment!
echo 🔮 Sit back, relax, and let Windows do its thing...
echo.

REM Check if Python is installed - Snake hunting time! 🐍
echo 🐍 Checking Python environment (Snake charming time)...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found! (No snake in sight!)
    echo 💡 Please install Python 3.8 or higher first
    echo 📦 Download from: https://www.python.org/downloads/
    echo 🎯 Pro tip: Check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python found: %PYTHON_VERSION% (Looking good!)

REM Check pip availability
echo 🔧 Checking system dependencies (Making sure everything's in place)...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not installed or unavailable
    echo 💡 Please install pip: python -m ensurepip --upgrade
    pause
    exit /b 1
)
echo ✅ pip available (Package manager ready!)

REM Check if setup script exists
echo 📁 Checking necessary files (Inventory time)...
if not exist "setup_auto_venv.py" (
    echo ❌ Missing setup script: setup_auto_venv.py
    echo 😱 The magic wand is missing!
    pause
    exit /b 1
)
echo ✅ Setup script found (Magic wand ready!)

if not exist "requirements.txt" (
    echo ❌ Missing requirements file: requirements.txt
    echo 📋 Shopping list not found!
    pause
    exit /b 1
)
echo ✅ Requirements file found (Shopping list ready!)

REM Clean up old environment
echo 🧹 Cleaning up old environment (Spring cleaning time)...
if exist "dl_course_env" (
    echo 🗑️ Removing old virtual environment (Out with the old)...
    rmdir /s /q dl_course_env
)
echo ✅ Environment cleanup complete (Squeaky clean!)

echo.
echo 🚀 Running environment setup script (The moment of truth)...
echo 📝 Watch the magic happen below!
echo 🍿 Grab some coffee - this might take a minute...
echo.

REM Run Python setup script
python setup_auto_venv.py

REM Check execution result
if errorlevel 1 (
    echo.
    echo ❌ Environment setup failed!
    echo 😞 Houston, we have a problem...
    echo 💡 Possible solutions:
    echo    1. Check your internet connection
    echo    2. Ensure Python version ^>= 3.8
    echo    3. Try running this script again
    echo    4. Check the generated log files
    echo    5. Take a coffee break and try again ☕
    pause
    exit /b 1
) else (
    echo.
    echo 🎉 Environment setup completed successfully!
    echo 🎊 You're ready to conquer deep learning!
    
    REM Additional NumPy compatibility check and fix
    echo 🔧 Applying NumPy compatibility fixes...
    call dl_course_env\Scripts\activate.bat
    pip install "numpy>=1.21.0,<2.0.0" --upgrade
    call deactivate
    echo ✅ NumPy compatibility ensured!
    
    echo.
    echo 🎯 Next steps (Your mission, should you choose to accept it):
    echo    1. Double-click activate_env.bat to activate environment
    echo    2. Run 'jupyter lab' to start Jupyter Lab
    echo    3. Open course notebooks and start your learning journey!
    echo.
    echo 📚 View detailed usage guide: ENVIRONMENT_GUIDE.md
    echo.
    echo 🔧 Environment Summary:
    echo    Virtual Environment: dl_course_env
    echo    Python Version: %PYTHON_VERSION%
    echo    Jupyter Kernel: Deep Learning Course
    echo    NumPy Version: Fixed for PyTorch compatibility
    echo    Status: Ready for action! 🚀
)

echo.
echo 🎓 Happy learning! May your gradients descend smoothly!
pause 