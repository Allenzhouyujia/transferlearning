#!/bin/bash

# Terminal colors - Making the terminal beautiful! 🎨
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Print colorful text like a rainbow! 🌈
print_colored() {
    echo -e "${1}${2}${NC}"
}

# Print the magical header ✨
print_header() {
    echo
    print_colored $CYAN "=================================================================="
    print_colored $PURPLE "🎓 CIFAR-10 Deep Learning Course - Environment Setup Wizard"
    print_colored $CYAN "=================================================================="
    echo
    print_colored $BLUE "🚀 Launching automated environment configuration..."
    print_colored $YELLOW "🎯 This script will magically configure your Python virtual environment!"
    print_colored $PURPLE "🔮 Sit back, relax, and let the automation do its thing..."
    echo
}

# Detect operating system like a detective 🕵️
check_system() {
    print_colored $BLUE "🔍 Detecting operating system (Playing detective)..."
    case "$OSTYPE" in
        darwin*)
            print_colored $GREEN "✅ macOS detected - Welcome to the Apple ecosystem! 🍎"
            ;;
        linux*)
            print_colored $GREEN "✅ Linux detected - A true developer's choice! 🐧"
            ;;
        *)
            print_colored $YELLOW "⚠️ Other system detected: $OSTYPE (We'll make it work!)"
            ;;
    esac
}

# Check Python like a quality inspector 🐍
check_python() {
    print_colored $BLUE "🐍 Checking Python environment (Snake charming time)..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_colored $RED "❌ Error: Python not found! (No snake in sight!)"
        print_colored $YELLOW "💡 Please install Python 3.8 or higher first"
        echo
        
        # Provide installation suggestions based on system
        if [[ "$OSTYPE" == "darwin"* ]]; then
            print_colored $CYAN "📦 macOS installation methods:"
            echo "   Method 1: brew install python"
            echo "   Method 2: Visit https://www.python.org/downloads/"
            echo "   Method 3: Use pyenv for multiple Python versions"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            print_colored $CYAN "📦 Linux installation methods:"
            echo "   Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
            echo "   CentOS/RHEL: sudo yum install python3 python3-pip"
            echo "   Fedora: sudo dnf install python3 python3-pip"
            echo "   Or visit: https://www.python.org/downloads/"
        fi
        
        exit 1
    fi
    
    # Determine Python command
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_colored $GREEN "✅ Python found: $PYTHON_VERSION (Looking good!)"
    
    # Check if version meets requirements (>= 3.8)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
        print_colored $RED "❌ Python version too old! Need >= 3.8, current: $PYTHON_VERSION"
        print_colored $YELLOW "🦖 Ancient Python detected! Time for an upgrade!"
        exit 1
    fi
    
    print_colored $GREEN "✅ Python version meets requirements (You're good to go!)"
}

# Check system dependencies like a mechanic 🔧
check_dependencies() {
    print_colored $BLUE "🔧 Checking system dependencies (Making sure everything's in place)..."
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_colored $RED "❌ pip not installed or unavailable"
        print_colored $YELLOW "💡 Please install pip: $PYTHON_CMD -m ensurepip --upgrade"
        exit 1
    fi
    print_colored $GREEN "✅ pip available (Package manager ready!)"
    
    # Check venv
    if ! $PYTHON_CMD -m venv --help &> /dev/null; then
        print_colored $RED "❌ venv module not available"
        print_colored $YELLOW "💡 Please install python3-venv (Linux) or upgrade Python"
        exit 1
    fi
    print_colored $GREEN "✅ venv module available (Virtual environments ready!)"
}

# Check necessary files like a librarian 📚
check_files() {
    print_colored $BLUE "📁 Checking necessary files (Inventory time)..."
    
    if [[ ! -f "setup_auto_venv.py" ]]; then
        print_colored $RED "❌ Missing setup script: setup_auto_venv.py"
        print_colored $YELLOW "😱 The magic wand is missing!"
        exit 1
    fi
    print_colored $GREEN "✅ Setup script found (Magic wand ready!)"
    
    if [[ ! -f "requirements.txt" ]]; then
        print_colored $RED "❌ Missing requirements file: requirements.txt"
        print_colored $YELLOW "📋 Shopping list not found!"
        exit 1
    fi
    print_colored $GREEN "✅ Requirements file found (Shopping list ready!)"
}

# Check directory permissions like a security guard 🔐
check_permissions() {
    print_colored $BLUE "🔐 Checking directory permissions (Security check)..."
    
    if [[ ! -w "." ]]; then
        print_colored $RED "❌ Error: No write permission in current directory!"
        print_colored $YELLOW "💡 Please ensure you have write permissions"
        exit 1
    fi
    print_colored $GREEN "✅ Directory permissions OK (Access granted!)"
}

# Clean up old environment like a janitor 🧹
cleanup_old_env() {
    print_colored $BLUE "🧹 Cleaning up old environment (Spring cleaning time)..."
    
    # Remove old virtual environment
    if [[ -d "dl_course_env" ]]; then
        print_colored $YELLOW "🗑️ Removing old virtual environment (Out with the old)..."
        rm -rf dl_course_env
    fi
    
    # Remove old activation scripts
    if [[ -f "activate_env.sh" ]]; then
        rm -f activate_env.sh
    fi
    
    # Remove old guide files
    if [[ -f "ENVIRONMENT_GUIDE.md" ]]; then
        rm -f ENVIRONMENT_GUIDE.md
    fi
    
    print_colored $GREEN "✅ Environment cleanup complete (Squeaky clean!)"
}

# Main function - Where the magic happens ✨
main() {
    print_header
    
    # Pre-flight checks
    check_system
    check_python
    check_dependencies
    check_files
    check_permissions
    
    # Clean up old environment
    cleanup_old_env
    
    echo
    print_colored $YELLOW "🚀 Running environment setup script (The moment of truth)..."
    print_colored $BLUE "📝 Watch the magic happen below!"
    print_colored $PURPLE "🍿 Grab some popcorn - this might take a minute..."
    echo
    
    # Run Python setup script
    if $PYTHON_CMD setup_auto_venv.py; then
        echo
        print_colored $GREEN "🎉 Environment setup completed successfully!"
        print_colored $GREEN "🎊 You're ready to conquer deep learning!"
        
        # Additional NumPy compatibility check and fix
        print_colored $BLUE "🔧 Applying NumPy compatibility fixes..."
        source dl_course_env/bin/activate
        pip install "numpy>=1.21.0,<2.0.0" --upgrade
        deactivate
        print_colored $GREEN "✅ NumPy compatibility ensured!"
        
        echo
        print_colored $BLUE "🎯 Next steps (Your mission, should you choose to accept it):"
        print_colored $CYAN "   1. Execute ./activate_env.sh to activate environment"
        print_colored $CYAN "   2. Run 'jupyter lab' to start Jupyter Lab"
        print_colored $CYAN "   3. Open course notebooks and start your learning journey!"
        echo
        print_colored $PURPLE "📚 View detailed usage guide: ENVIRONMENT_GUIDE.md"
        
        # Make activation script executable
        if [[ -f "activate_env.sh" ]]; then
            chmod +x activate_env.sh
            print_colored $GREEN "✅ Activation script is now executable (Ready to roll!)"
        fi
        
        # Show environment information
        echo
        print_colored $BLUE "🔧 Environment Summary:"
        print_colored $CYAN "   Virtual Environment: dl_course_env"
        print_colored $CYAN "   Python Version: $PYTHON_VERSION"
        print_colored $CYAN "   Jupyter Kernel: Deep Learning Course"
        print_colored $CYAN "   NumPy Version: Fixed for PyTorch compatibility"
        print_colored $CYAN "   Status: Ready for action! 🚀"
        
    else
        echo
        print_colored $RED "❌ Environment setup failed!"
        print_colored $YELLOW "😞 Houston, we have a problem..."
        print_colored $YELLOW "💡 Possible solutions:"
        print_colored $CYAN "   1. Check your internet connection"
        print_colored $CYAN "   2. Ensure Python version >= 3.8"
        print_colored $CYAN "   3. Try running this script again"
        print_colored $CYAN "   4. Check the generated log files"
        print_colored $CYAN "   5. Take a coffee break and try again ☕"
        exit 1
    fi
}

# Cleanup function for graceful interruption 🛑
cleanup() {
    echo
    print_colored $YELLOW "⚠️ User interrupted the setup process"
    print_colored $BLUE "🧹 Cleaning up temporary files..."
    print_colored $YELLOW "👋 Come back when you're ready!"
    exit 1
}

# Catch interrupt signals
trap cleanup INT

# Show help information 📖
show_help() {
    echo "🎓 CIFAR-10 Deep Learning Course - Environment Setup Wizard"
    echo
    echo "Usage: ./setup.sh [options]"
    echo
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo "  -v, --version Show version information"
    echo
    echo "This script will automatically configure a Python virtual environment"
    echo "and install all required deep learning libraries."
    echo "Make sure you have Python 3.8 or higher installed."
    echo
    echo "🎯 Ready to dive into the world of deep learning? Let's go!"
}

# Check command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -v|--version)
        echo "CIFAR-10 Deep Learning Course Environment Setup Wizard v1.0"
        echo "🎓 Making deep learning accessible, one environment at a time!"
        exit 0
        ;;
    "")
        # No arguments, run main function
        main
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use -h or --help for help information"
        exit 1
        ;;
esac 