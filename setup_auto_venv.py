#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎓 CIFAR-10 Deep Learning Course - Automated Environment Setup
🚀 An intelligent cross-platform environment configurator

This script automatically sets up a complete deep learning environment
with PyTorch, data science libraries, and Jupyter Lab integration.

Author: AI Assistant
Version: 1.0.0
"""

import os
import sys
import platform
import subprocess
import shutil
import time
from pathlib import Path
import json

# Configuration constants
VENV_NAME = "dl_course_env"
REQUIREMENTS_FILE = "requirements.txt"
PROJECT_NAME = "CIFAR-10 Deep Learning Course"
JUPYTER_KERNEL_NAME = "dl_course"
JUPYTER_DISPLAY_NAME = "Deep Learning Course"

class Colors:
    """Terminal color constants for beautiful output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    PURPLE = '\033[95m'

class SystemEnvironmentSetup:
    """Cross-platform environment setup wizard 🧙‍♂️"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / VENV_NAME
        self.success_log = []
        self.error_log = []
        
    def print_colored(self, message, color=Colors.ENDC):
        """Print colorful messages to brighten your day! 🌈"""
        print(f"{color}{message}{Colors.ENDC}")
        
    def print_header(self):
        """Print the magical header with style ✨"""
        self.print_colored("=" * 70, Colors.CYAN)
        self.print_colored(f"🎓 {PROJECT_NAME} - Environment Setup Wizard", Colors.HEADER + Colors.BOLD)
        self.print_colored("=" * 70, Colors.CYAN)
        print()
        self.print_colored("🎯 This script will magically configure your deep learning environment!", Colors.BLUE)
        self.print_colored("🔮 Sit back and let the automation do the work...", Colors.PURPLE)
        print()
        
    def detect_system(self):
        """Detect system information like a detective 🕵️‍♂️"""
        self.print_colored("🔍 Detecting system environment (CSI: Computer Edition)...", Colors.BLUE)
        
        # Basic system info
        system_info = {
            "Operating System": platform.system(),
            "System Version": platform.release(),
            "Architecture": platform.machine(),
            "Python Version": platform.python_version(),
            "Python Path": sys.executable
        }
        
        print()
        for key, value in system_info.items():
            self.print_colored(f"   {key}: {value}", Colors.CYAN)
        print()
        
        # Special detection with fun messages
        if self.system == "darwin":
            self.print_colored("🍎 macOS detected - Welcome to the Apple ecosystem!", Colors.GREEN)
            # Apple Silicon detection
            if platform.machine() == "arm64":
                self.print_colored("   🚀 Apple Silicon (M1/M2) detected - You've got some serious power!", Colors.GREEN)
            else:
                self.print_colored("   💻 Intel chip detected - Classic and reliable!", Colors.GREEN)
                
        elif self.system == "windows":
            self.print_colored("🪟 Windows detected - Let's make this work seamlessly!", Colors.GREEN)
            
        elif self.system == "linux":
            self.print_colored("🐧 Linux detected - A true developer's choice!", Colors.GREEN)
            # Detect distribution
            try:
                with open('/etc/os-release', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('PRETTY_NAME'):
                            distro = line.split('=')[1].strip().strip('"')
                            self.print_colored(f"   📦 Distribution: {distro}", Colors.GREEN)
                            break
            except:
                pass
        
        # Python version check with humor
        min_version = (3, 8)
        current_version = (sys.version_info.major, sys.version_info.minor)
        
        if current_version >= min_version:
            self.print_colored(f"✅ Python version check passed ({self.python_version}) - You're good to go!", Colors.GREEN)
            self.success_log.append("Python version meets requirements")
        else:
            self.print_colored(f"❌ Python version too old! Need >= {min_version[0]}.{min_version[1]} (Ancient Python detected!)", Colors.RED)
            self.error_log.append(f"Python version insufficient: {self.python_version}")
            return False
            
        return True
    
    def check_dependencies(self):
        """Check system dependencies like a quality inspector 🔧"""
        self.print_colored("\n🔧 Checking system dependencies (Making sure everything's in place)...", Colors.BLUE)
        
        # Check pip
        try:
            import pip
            self.print_colored("   ✅ pip is installed - The package manager is ready!", Colors.GREEN)
            self.success_log.append("pip available")
        except ImportError:
            self.print_colored("   ❌ pip not installed! Please install pip first", Colors.RED)
            self.error_log.append("pip not installed")
            return False
        
        # Check venv module
        try:
            import venv
            self.print_colored("   ✅ venv module available - Virtual environments ready!", Colors.GREEN)
            self.success_log.append("venv module available")
        except ImportError:
            self.print_colored("   ❌ venv module not available!", Colors.RED)
            self.error_log.append("venv module not available")
            return False
            
        # Check network connection
        self.print_colored("   🌐 Checking network connection (Reaching out to the internet)...", Colors.YELLOW)
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org', timeout=10)
            self.print_colored("   ✅ Network connection is healthy - Ready to download packages!", Colors.GREEN)
            self.success_log.append("Network connection healthy")
        except:
            self.print_colored("   ⚠️ Network might be slow, but we'll persevere!", Colors.YELLOW)
            
        return True
    
    def create_virtual_environment(self):
        """Create virtual environment like building a fortress 🏰"""
        self.print_colored(f"\n🏗️ Creating virtual environment: {VENV_NAME} (Building your coding fortress)", Colors.BLUE)
        
        # Remove existing environment
        if self.venv_path.exists():
            self.print_colored(f"   🗑️ Removing old environment (Out with the old)...", Colors.YELLOW)
            shutil.rmtree(self.venv_path)
            
        try:
            # Create virtual environment
            self.print_colored(f"   🔨 Creating virtual environment (Crafting your isolated paradise)...", Colors.YELLOW)
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True, capture_output=True)
            
            self.print_colored(f"   ✅ Virtual environment created successfully: {self.venv_path}", Colors.GREEN)
            self.success_log.append("Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_colored(f"   ❌ Virtual environment creation failed: {e}", Colors.RED)
            self.error_log.append(f"Virtual environment creation failed: {e}")
            return False
    
    def get_venv_python(self):
        """Get Python path in virtual environment"""
        if self.system == "windows":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def get_venv_pip(self):
        """Get pip path in virtual environment"""
        if self.system == "windows":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def upgrade_pip(self):
        """Upgrade pip to latest version (Getting the freshest pip!) 📦"""
        self.print_colored("\n📦 Upgrading pip to latest version (Fresher than morning coffee)...", Colors.BLUE)
        
        try:
            venv_pip = self.get_venv_pip()
            subprocess.run([
                str(venv_pip), "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            self.print_colored("   ✅ pip upgraded successfully - Now with extra features!", Colors.GREEN)
            self.success_log.append("pip upgraded successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_colored(f"   ❌ pip upgrade failed: {e}", Colors.RED)
            self.error_log.append(f"pip upgrade failed: {e}")
            return False
    
    def install_requirements(self):
        """Install project dependencies (Time for the heavy lifting!) 💪"""
        self.print_colored("\n📥 Installing project dependencies (The moment of truth)...", Colors.BLUE)
        
        if not Path(REQUIREMENTS_FILE).exists():
            self.print_colored(f"   ❌ Requirements file not found: {REQUIREMENTS_FILE}", Colors.RED)
            self.error_log.append(f"Requirements file not found: {REQUIREMENTS_FILE}")
            return False
            
        try:
            venv_pip = self.get_venv_pip()
            
            # Fix NumPy compatibility issue first
            self.print_colored("   🔧 Ensuring NumPy compatibility (Fixing known issues)...", Colors.YELLOW)
            subprocess.run([
                str(venv_pip), "install", "numpy>=1.21.0,<2.0.0"
            ], check=True, capture_output=True)
            
            self.print_colored("   📦 Installing dependency packages (Downloading the internet)...", Colors.YELLOW)
            subprocess.run([
                str(venv_pip), "install", "-r", REQUIREMENTS_FILE
            ], check=True, capture_output=True)
            
            self.print_colored("   ✅ Dependencies installed successfully - Ready to rock!", Colors.GREEN)
            self.success_log.append("Dependencies installed successfully")
            return True
                
        except subprocess.CalledProcessError as e:
            self.print_colored(f"   ❌ Dependency installation failed: {e}", Colors.RED)
            self.error_log.append(f"Dependency installation failed: {e}")
            return False
    
    def setup_jupyter_kernel(self):
        """Setup Jupyter kernel (Making Jupyter Lab beautiful!) 🎨"""
        self.print_colored("\n🔬 Configuring Jupyter kernel (Setting up your coding playground)...", Colors.BLUE)
        
        try:
            venv_python = self.get_venv_python()
            
            # Install ipykernel
            self.print_colored("   📦 Installing ipykernel (The bridge to Jupyter)...", Colors.YELLOW)
            subprocess.run([
                str(venv_python), "-m", "pip", "install", "ipykernel"
            ], check=True, capture_output=True)
            
            # Add Jupyter kernel
            self.print_colored("   🔧 Adding Jupyter kernel (Making it official)...", Colors.YELLOW)
            subprocess.run([
                str(venv_python), "-m", "ipykernel", "install", "--user", 
                "--name", JUPYTER_KERNEL_NAME, "--display-name", JUPYTER_DISPLAY_NAME
            ], check=True, capture_output=True)
            
            self.print_colored(f"   ✅ Jupyter kernel configured successfully: {JUPYTER_DISPLAY_NAME}", Colors.GREEN)
            self.success_log.append("Jupyter kernel configured successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_colored(f"   ❌ Jupyter kernel setup failed: {e}", Colors.RED)
            self.error_log.append(f"Jupyter kernel setup failed: {e}")
            return False
    
    def verify_installation(self):
        """Verify installation like a final exam 🎓"""
        self.print_colored("\n🧪 Verifying installation (Final boss battle)...", Colors.BLUE)
        
        try:
            venv_python = self.get_venv_python()
            
            # Test imports
            test_imports = [
                ("torch", "PyTorch - The deep learning powerhouse"),
                ("torchvision", "Torchvision - Computer vision made easy"),
                ("numpy", "NumPy - The numerical computing foundation"),
                ("matplotlib", "Matplotlib - Your plotting best friend"),
                ("jupyter", "Jupyter - The interactive computing star")
            ]
            
            for module, description in test_imports:
                result = subprocess.run([
                    str(venv_python), "-c", f"import {module}"
                ], capture_output=True)
                
                if result.returncode == 0:
                    self.print_colored(f"   ✅ {description}", Colors.GREEN)
                else:
                    self.print_colored(f"   ❌ {module} import failed", Colors.RED)
                    self.error_log.append(f"{module} import failed")
            
            self.success_log.append("Installation verification completed")
            return True
            
        except Exception as e:
            self.print_colored(f"   ❌ Verification process failed: {e}", Colors.RED)
            return False
    
    def generate_activation_scripts(self):
        """Generate activation scripts (Your keys to the kingdom!) 🗝️"""
        self.print_colored(f"\n📝 Generating activation scripts (Creating your magic wands)...", Colors.BLUE)
        
        # Windows batch script
        if self.system == "windows":
            bat_content = f"""@echo off
echo 🔧 Activating deep learning environment...
echo 🎯 Welcome to your coding sanctuary!
call "{self.venv_path}\\Scripts\\activate.bat"
echo ✅ Environment activated successfully!
echo 💡 Use 'jupyter lab' to start Jupyter Lab
echo 💡 Use 'deactivate' to exit the environment
cmd /k
"""
            with open("activate_env.bat", "w", encoding="utf-8") as f:
                f.write(bat_content)
            self.print_colored("   ✅ Windows activation script: activate_env.bat", Colors.GREEN)
        
        # Unix shell script
        shell_content = f"""#!/bin/bash
echo "🔧 Activating deep learning environment..."
echo "🎯 Welcome to your coding sanctuary!"
source "{self.venv_path}/bin/activate"
echo "✅ Environment activated successfully!"
echo "💡 Use 'jupyter lab' to start Jupyter Lab"
echo "💡 Use 'deactivate' to exit the environment"
exec "$SHELL"
"""
        with open("activate_env.sh", "w") as f:
            f.write(shell_content)
        os.chmod("activate_env.sh", 0o755)
        self.print_colored("   ✅ Unix activation script: activate_env.sh", Colors.GREEN)
        
        self.success_log.append("Activation scripts generated successfully")
    
    def create_usage_guide(self):
        """Create usage guide (Your roadmap to success!) 🗺️"""
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

        guide_content = f"""# 🎓 {PROJECT_NAME} Usage Guide

## 🚀 Quick Start

### Method 1: Use Activation Scripts
"""
        
        if self.system == "windows":
            guide_content += """
**Windows:**
```cmd
# Double-click to run or execute in command line
activate_env.bat
```
"""
        
        guide_content += f"""
**macOS/Linux:**
```bash
# Execute in terminal
./activate_env.sh
```

### Method 2: Manual Activation

**Windows:**
```cmd
{self.venv_path}\\Scripts\\activate
```

**macOS/Linux:**
```bash
source {self.venv_path}/bin/activate
```

## 📚 Start Learning

After activating the environment, you can:

1. **Launch Jupyter Lab**
   ```bash
   jupyter lab
   ```

2. **Run Course Lessons**
   - 📊 Lesson 2: Data Exploration
     ```bash
     jupyter lab lessons/lesson2_data_exploration/data_exploration.ipynb
     ```

3. **Verify Environment**
   ```bash
   python lessons/lesson0_environment_setup/main.py
   ```

## 🔧 Environment Management

- **Activate Environment**: Use the activation methods above
- **Exit Environment**: Type `deactivate` in the activated environment
- **Reconfigure**: Re-run `python setup_auto_venv.py`

## 🎯 Jupyter Kernel

The environment has a dedicated Jupyter kernel configured:
- Kernel Name: {JUPYTER_KERNEL_NAME}
- Display Name: {JUPYTER_DISPLAY_NAME}

Select this kernel in Jupyter to run course code.

## 🆘 Troubleshooting

If you encounter issues:
1. Ensure Python version >= 3.8
2. Check network connection
3. Re-run the environment setup script
4. Check the generated log files

---
*Auto-generated on: {current_time}*
"""
        
        with open("ENVIRONMENT_GUIDE.md", "w", encoding="utf-8") as f:
            f.write(guide_content)
        
        self.print_colored("   ✅ Usage guide created: ENVIRONMENT_GUIDE.md", Colors.GREEN)
    
    def save_setup_log(self):
        """Save installation log for posterity 📝"""
        log_data = {
            "setup_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "system_info": {
                "os": platform.system(),
                "version": platform.release(),
                "architecture": platform.machine(),
                "python_version": platform.python_version()
            },
            "venv_path": str(self.venv_path),
            "jupyter_kernel": JUPYTER_KERNEL_NAME,
            "success_log": self.success_log,
            "error_log": self.error_log
        }
        
        with open("setup_log.json", "w") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def print_final_summary(self):
        """Print final summary with style! 🎉"""
        self.print_colored("\n" + "="*70, Colors.CYAN)
        
        if len(self.error_log) == 0:
            self.print_colored("🎉 Environment setup completed successfully!", Colors.GREEN + Colors.BOLD)
            self.print_colored("🎊 You're ready to dive into deep learning!", Colors.GREEN)
            self.print_colored("\n✅ All components installed successfully:", Colors.GREEN)
            for item in self.success_log:
                self.print_colored(f"   ✓ {item}", Colors.GREEN)
        else:
            self.print_colored("⚠️ Environment setup completed with some issues:", Colors.YELLOW + Colors.BOLD)
            self.print_colored("\n❌ Issues encountered:", Colors.RED)
            for item in self.error_log:
                self.print_colored(f"   ✗ {item}", Colors.RED)
                
        self.print_colored(f"\n🎯 Next Steps:", Colors.BLUE + Colors.BOLD)
        
        if self.system == "windows":
            self.print_colored("   1. Double-click activate_env.bat to activate environment", Colors.CYAN)
        else:
            self.print_colored("   1. Execute ./activate_env.sh to activate environment", Colors.CYAN)
            
        self.print_colored("   2. Run 'jupyter lab' to start Jupyter Lab", Colors.CYAN)
        self.print_colored("   3. Open course notebooks and start learning!", Colors.CYAN)
        
        self.print_colored(f"\n📚 View detailed usage guide: ENVIRONMENT_GUIDE.md", Colors.BLUE)
        self.print_colored("🎓 Happy learning! May your gradients descend smoothly!", Colors.PURPLE)
        self.print_colored("="*70, Colors.CYAN)
    
    def run_setup(self):
        """Run the complete environment setup wizard 🧙‍♂️"""
        self.print_header()
        
        # Step 1: System detection
        if not self.detect_system():
            return False
            
        # Step 2: Check dependencies
        if not self.check_dependencies():
            return False
            
        # Step 3: Create virtual environment
        if not self.create_virtual_environment():
            return False
            
        # Step 4: Upgrade pip
        self.upgrade_pip()
        
        # Step 5: Install dependencies
        if not self.install_requirements():
            return False
            
        # Step 6: Configure Jupyter
        self.setup_jupyter_kernel()
        
        # Step 7: Verify installation
        self.verify_installation()
        
        # Step 8: Generate scripts and guides
        self.generate_activation_scripts()
        self.create_usage_guide()
        
        # Step 9: Save log
        self.save_setup_log()
        
        # Step 10: Print summary
        self.print_final_summary()
        
        return True

def main():
    """Main function - Let the magic begin! ✨"""
    try:
        setup = SystemEnvironmentSetup()
        success = setup.run_setup()
        
        if success and len(setup.error_log) == 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ User interrupted the setup process{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Unexpected error during setup: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main() 