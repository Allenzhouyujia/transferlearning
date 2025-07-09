#!/usr/bin/env python3
"""
🎓 Lesson 0: Environment Setup & Virtual Environment Configuration
Interactive Setup Check Script

This script provides comprehensive environment verification for deep learning.
Run this if you're having issues with the Jupyter notebook version.

Usage:
    python interactive_setup_check.py
"""

import sys
import os
import platform
import subprocess
from datetime import datetime

def print_header():
    """Print the course header"""
    print("🎓 LESSON 0: ENVIRONMENT SETUP & CONFIGURATION")
    print("=" * 60)
    print("Welcome to the foundation of our deep learning journey!")
    print("This script will verify your environment setup and ensure")
    print("everything is working correctly for upcoming lessons.")
    print()
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def check_imports():
    """Check if all required libraries can be imported"""
    print("\n🔧 STEP 1: CHECKING LIBRARY IMPORTS")
    print("=" * 60)
    
    imports_status = {}
    
    # Core libraries
    libraries = [
        ('sys', 'sys'),
        ('os', 'os'),
        ('platform', 'platform'),
        ('datetime', 'datetime'),
        ('torch', 'torch'),
        ('torch.nn', 'torch.nn'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib.pyplot', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('PIL', 'PIL'),
        ('tqdm', 'tqdm'),
    ]
    
    for lib_name, import_name in libraries:
        try:
            __import__(import_name)
            print(f"✅ {lib_name}: Successfully imported")
            imports_status[lib_name] = True
        except ImportError as e:
            print(f"❌ {lib_name}: Failed to import - {e}")
            imports_status[lib_name] = False
    
    successful_imports = sum(imports_status.values())
    total_imports = len(imports_status)
    
    print(f"\n📊 Import Results: {successful_imports}/{total_imports} successful")
    
    if successful_imports == total_imports:
        print("🎉 All libraries imported successfully!")
        return True
    else:
        print("⚠️  Some libraries failed to import. Please install missing packages.")
        return False

def check_system_info():
    """Check system information"""
    print("\n🖥️ STEP 2: SYSTEM INFORMATION")
    print("=" * 60)
    
    info = {
        'System': platform.system(),
        'Release': platform.release(),
        'Machine': platform.machine(),
        'Processor': platform.processor(),
        'Python Version': sys.version.split()[0],
        'Python Executable': sys.executable,
        'Current Directory': os.getcwd(),
    }
    
    for key, value in info.items():
        print(f"📋 {key}: {value}")
    
    # Virtual environment check
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"✅ Virtual Environment: {os.path.basename(venv)}")
        print("🎯 Running in virtual environment - Great!")
    else:
        print("⚠️  Virtual Environment: Not active")
        print("💡 Consider using a virtual environment for better dependency management")
    
    return venv is not None

def check_hardware():
    """Check hardware acceleration capabilities"""
    print("\n🎮 STEP 3: HARDWARE ACCELERATION CHECK")
    print("=" * 60)
    
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            print("✅ CUDA (NVIDIA GPU) is available!")
            print(f"   🎯 CUDA Version: {torch.version.cuda}")
            print(f"   🎯 Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   🎯 GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   🎯 Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            device = torch.device("cuda")
        else:
            print("❌ CUDA (NVIDIA GPU) is not available")
        
        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon GPU) is available!")
            print("   🎯 You can use Apple's Metal Performance Shaders")
            device = torch.device("mps") if not torch.cuda.is_available() else device
        else:
            print("❌ MPS (Apple Silicon GPU) is not available")
        
        # Set device
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            device = torch.device("cpu")
            print("🖥️ Using CPU for computation")
        
        print(f"\n🚀 Recommended device: {device}")
        return device
        
    except ImportError:
        print("❌ PyTorch not available - cannot check hardware acceleration")
        return None

def check_package_versions():
    """Check versions of important packages"""
    print("\n📦 STEP 4: PACKAGE VERSIONS")
    print("=" * 60)
    
    packages = []
    
    try:
        import torch
        packages.append(('torch', torch.__version__))
    except ImportError:
        packages.append(('torch', 'NOT INSTALLED'))
    
    try:
        import torchvision
        packages.append(('torchvision', torchvision.__version__))
    except ImportError:
        packages.append(('torchvision', 'NOT INSTALLED'))
    
    try:
        import numpy
        packages.append(('numpy', numpy.__version__))
    except ImportError:
        packages.append(('numpy', 'NOT INSTALLED'))
    
    try:
        import pandas
        packages.append(('pandas', pandas.__version__))
    except ImportError:
        packages.append(('pandas', 'NOT INSTALLED'))
    
    try:
        import matplotlib
        packages.append(('matplotlib', matplotlib.__version__))
    except ImportError:
        packages.append(('matplotlib', 'NOT INSTALLED'))
    
    try:
        import seaborn
        packages.append(('seaborn', seaborn.__version__))
    except ImportError:
        packages.append(('seaborn', 'NOT INSTALLED'))
    
    # Display package versions
    for package, version in packages:
        status = "✅" if version != 'NOT INSTALLED' else "❌"
        print(f"{status} {package}: {version}")
    
    return packages

def run_functional_tests(device):
    """Run functional tests"""
    print("\n🧪 STEP 5: FUNCTIONAL TESTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        
        # Test 1: Basic tensor operations
        total_tests += 1
        try:
            x = torch.randn(3, 3, device=device)
            y = torch.randn(3, 3, device=device)
            z = torch.mm(x, y)
            assert z.shape == (3, 3)
            print("✅ Test 1: Basic tensor operations - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 1: Basic tensor operations - FAILED: {e}")
        
        # Test 2: Neural network
        total_tests += 1
        try:
            model = nn.Linear(10, 5).to(device)
            input_tensor = torch.randn(2, 10, device=device)
            output = model(input_tensor)
            assert output.shape == (2, 5)
            print("✅ Test 2: Neural network - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 2: Neural network - FAILED: {e}")
        
        # Test 3: Gradient computation
        total_tests += 1
        try:
            x = torch.randn(2, 2, device=device, requires_grad=True)
            y = x.sum()
            y.backward()
            assert x.grad is not None
            print("✅ Test 3: Gradient computation - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 3: Gradient computation - FAILED: {e}")
        
        # Test 4: NumPy operations
        total_tests += 1
        try:
            arr = np.random.randn(100, 100)
            result = np.mean(arr)
            assert isinstance(result, (float, np.floating))
            print("✅ Test 4: NumPy operations - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 4: NumPy operations - FAILED: {e}")
        
        # Test 5: Basic visualization
        total_tests += 1
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            plt.close(fig)  # Close to avoid display issues
            print("✅ Test 5: Basic visualization - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 5: Basic visualization - FAILED: {e}")
        
    except ImportError:
        print("❌ Required libraries not available for testing")
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    success_rate = (tests_passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    return tests_passed, total_tests

def generate_summary(imports_ok, venv_active, device, tests_passed, total_tests):
    """Generate comprehensive summary"""
    print("\n🎉 ENVIRONMENT SETUP SUMMARY")
    print("=" * 60)
    
    # Environment status
    if venv_active:
        print("✅ Virtual Environment: Active")
    else:
        print("⚠️  Virtual Environment: Not active")
    
    # Import status
    if imports_ok:
        print("✅ Library Imports: All successful")
    else:
        print("❌ Library Imports: Some failed")
    
    # Hardware status
    if device:
        if device.type == 'cuda':
            print(f"✅ Hardware: CUDA GPU available")
        elif device.type == 'mps':
            print("✅ Hardware: Apple Silicon GPU (MPS) available")
        else:
            print("✅ Hardware: CPU only")
    else:
        print("❌ Hardware: Could not detect")
    
    # Test results
    if tests_passed == total_tests and total_tests > 0:
        print(f"✅ Functional Tests: All {total_tests} tests passed")
    else:
        print(f"⚠️  Functional Tests: {tests_passed}/{total_tests} tests passed")
    
    print("\n" + "=" * 60)
    
    # Overall verdict
    if imports_ok and tests_passed == total_tests and total_tests > 0:
        if device and device.type in ['cuda', 'mps']:
            print("🚀 EXCELLENT! Your environment is perfectly configured!")
            print("⚡ Hardware acceleration is available for optimal performance!")
        else:
            print("✅ GOOD! Your environment is properly configured!")
            print("💡 Consider using GPU acceleration for better performance!")
        print("🎯 You're ready to start your deep learning journey!")
    else:
        print("⚠️  ATTENTION! Your environment needs some fixes.")
        print("🔧 Please address the failed checks above.")
    
    # Next steps
    print("\n📚 NEXT STEPS:")
    print("1. 📊 Lesson 2: CIFAR-10 Data Exploration")
    print("2. 🎯 Start with CIFAR-10 dataset analysis")
    print("3. 🧠 Begin building your first neural networks!")
    
    # Tips
    print("\n💡 TIPS:")
    if not venv_active:
        print("- Consider using a virtual environment for better dependency management")
    if device and device.type == 'cpu':
        print("- Consider using GPU acceleration for faster training")
    if not imports_ok:
        print("- Install missing packages using: pip install torch torchvision numpy pandas matplotlib seaborn pillow tqdm")
    
    print("\n🎓 Happy Learning!")
    print("=" * 60)

def main():
    """Main execution function"""
    print_header()
    
    # Run all checks
    imports_ok = check_imports()
    venv_active = check_system_info()
    device = check_hardware()
    
    if imports_ok:
        check_package_versions()
        tests_passed, total_tests = run_functional_tests(device)
    else:
        tests_passed, total_tests = 0, 0
    
    # Generate summary
    generate_summary(imports_ok, venv_active, device, tests_passed, total_tests)
    
    return imports_ok and tests_passed == total_tests

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup check interrupted by user.")
        print("👋 Come back when you're ready!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("🔧 Please check your environment and try again.")
        sys.exit(1) 