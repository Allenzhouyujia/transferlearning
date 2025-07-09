# 🎓 Lesson 0: Environment Setup & Virtual Environment Configuration

## 📋 Course Overview

This lesson serves as the foundation for our entire deep learning journey. We'll ensure your development environment is properly configured and optimized for all upcoming deep learning tasks.

### 🎯 Learning Objectives

By completing this lesson, you will:
- ✅ Have a dedicated Python virtual environment for deep learning
- ✅ Understand the core components of a deep learning development environment
- ✅ Master environment troubleshooting and problem-solving techniques
- ✅ Be fully prepared for upcoming Jupyter Notebook-based courses
- ✅ Have established performance benchmarks for your hardware

## ⚠️ Important Notes

### 🚨 Why Virtual Environments Are Essential

**Why must we use virtual environments?**

1. **Dependency Isolation**: Deep learning projects require specific package versions to avoid conflicts with system Python
2. **Version Management**: PyTorch, CUDA, and other tools have strict version compatibility requirements
3. **Jupyter Support**: Subsequent courses use `.ipynb` files that need to run in the virtual environment
4. **Project Portability**: Ensures consistent environments across different machines

### 📊 Upcoming Courses Use Jupyter Notebooks

Starting from Lesson 2, all data processing and model training courses use `.ipynb` format:

- **Lesson 2**: `data_exploration.ipynb` - CIFAR-10 data download and visualization
- **Lesson 3**: `basic_cnn_training.ipynb` - Traditional CNN training
- **Lesson 4**: `resnet_training.ipynb` - ResNet architecture training
- **Lesson 5**: `efficientnet_training.ipynb` - EfficientNet training
- **Lesson 6**: `mobilenet_training.ipynb` - MobileNet training
- **Lesson 7**: `model_comparison.ipynb` - Comprehensive model comparison

## 🛠️ Core Tools Introduction

### 🐍 Python Virtual Environment (venv)
```bash
python -m venv venv  # Creates isolated Python environment
```
**Purpose**: Creates independent Python environment, prevents package conflicts, ensures consistent project dependencies

### 🔥 PyTorch Deep Learning Framework
```python
import torch  # Core tensor computation library
import torchvision  # Computer vision toolkit
```
**Purpose**: 
- `torch`: Provides tensor computation, automatic differentiation, neural network building
- `torchvision`: Provides datasets, pre-trained models, image transformations

### 📊 Data Science Toolkit
```python
import numpy as np      # Numerical computation foundation
import matplotlib.pyplot as plt  # Basic plotting library
import seaborn as sns   # Statistical visualization library
import pandas as pd     # Data processing library
```
**Purpose**:
- `numpy`: High-performance numerical computation, tensor operation foundation
- `matplotlib`: Create charts, visualize training progress
- `seaborn`: Beautiful statistical charts, data analysis visualization
- `pandas`: Structured data processing, experimental results analysis

### 🧪 Machine Learning Support Tools
```python
from sklearn.metrics import classification_report  # Model evaluation
from tqdm import tqdm  # Progress bar display
```
**Purpose**:
- `scikit-learn`: Provides evaluation metrics, data preprocessing tools
- `tqdm`: Display training progress, enhance user experience

### 🎮 Hardware Acceleration Detection
```python
# CUDA (NVIDIA GPU)
torch.cuda.is_available()  # Detect NVIDIA GPU
torch.cuda.get_device_name(0)  # Get GPU name

# MPS (Apple Silicon)
torch.backends.mps.is_available()  # Detect Apple Silicon GPU
```
**Purpose**: Automatically detect available hardware acceleration, select optimal computing device

## 📁 File Structure

```
lesson0_environment_setup/
├── README.md                      # This file: Course description and tool introduction
├── main.py                        # Environment detection script: Automated diagnosis of all environment configurations
├── setup_check.ipynb              # Jupyter version: Interactive environment verification
└── interactive_setup_check.py     # Alternative Python script: Works if notebook has issues
```

## 🚀 Usage Methods

### Method 1: Interactive Python Script (Recommended if notebook has issues)
```bash
cd lessons/lesson0_environment_setup
python interactive_setup_check.py
```

### Method 2: Automated Script (Quick check)
```bash
cd lessons/lesson0_environment_setup
python main.py
```

### Method 3: Jupyter Notebook (Most comprehensive)
```bash
# Start Jupyter Lab
jupyter lab

# Open setup_check.ipynb
# Execute code blocks one by one for verification
```

## 💡 Troubleshooting the Notebook

If you're experiencing issues with `setup_check.ipynb`:

1. **Try the Interactive Script**: Use `interactive_setup_check.py` instead
2. **Check Jupyter Installation**: Ensure Jupyter is properly installed in your virtual environment
3. **Restart Jupyter**: Sometimes restarting Jupyter Lab helps
4. **Use the Automated Script**: The `main.py` script provides similar functionality

### Quick Fix Commands
```bash
# If notebook doesn't work, use the interactive script
python interactive_setup_check.py

# Or reinstall Jupyter in your virtual environment
pip install jupyter jupyterlab ipykernel
python -m ipykernel install --user --name=dl_course --display-name="Deep Learning Course"
```

## 🔍 Key Verification Items

### 1. Python Environment Verification
- **Python Version**: >= 3.8 (minimum requirement for deep learning frameworks)
- **Virtual Environment**: Ensure running in isolated environment
- **Package Manager**: pip version and functionality

### 2. Deep Learning Framework Detection
- **PyTorch Installation**: Version compatibility check
- **Hardware Support**: CUDA/MPS/CPU device detection
- **Basic Functions**: Tensor computation, gradient computation verification

### 3. Data Processing Tools Verification
- **Numerical Computation**: NumPy array operations
- **Data Visualization**: Matplotlib plotting functionality
- **Data Processing**: Pandas basic operations

### 4. Hardware Performance Benchmark
- **Computing Performance**: Matrix multiplication speed tests
- **Memory Usage**: GPU/system memory detection
- **I/O Performance**: Data loading speed tests

## 🎯 Success Criteria

After completing Lesson 0, you should see:

```
🎉 All environment checks passed!
✅ Python Environment: 3.8+ ✓
✅ PyTorch Installation: Normal ✓
✅ Hardware Acceleration: GPU/MPS available ✓
✅ Dependencies: All installed ✓
✅ Function Tests: Passed ✓
✅ Performance Benchmark: Established ✓

🎯 You're ready to start your deep learning journey!
📚 Next: Lesson 2 - CIFAR-10 Data Exploration
```

## 🔧 Troubleshooting

### ❌ Common Issues

#### 1. Virtual Environment Creation Failed
```bash
# Solution 1: Update pip
python -m pip install --upgrade pip

# Solution 2: Use conda
conda create -n dl_course python=3.9
conda activate dl_course
```

#### 2. PyTorch Installation Issues
```bash
# Check Python version
python --version

# Clear pip cache
pip cache purge

# Reinstall PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Jupyter Kernel Issues
```bash
# Install ipykernel
pip install ipykernel

# Add virtual environment to Jupyter
python -m ipykernel install --user --name=dl_course --display-name="Deep Learning Course"
```

## 📚 Next Steps

After environment configuration is complete, please continue with:

1. **📊 Lesson 2**: CIFAR-10 Data Exploration (`data_exploration.ipynb`)
2. **🏗️ Lesson 3**: Basic CNN Implementation (`basic_cnn_training.ipynb`)

---

**🎓 Happy Learning!** 