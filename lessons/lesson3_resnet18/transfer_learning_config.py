"""
Transfer Learning Configuration for ResNet18 - Lesson 3
Standardized hyperparameters for ResNet18 architecture on Flowers102 dataset
"""

# ResNet18 Model Configuration
MODEL_CONFIG = {
    'architecture': 'resnet18',
    'pretrained': True,
    'num_classes': 102,
    'total_params': '11.7M',
    'input_size': 224,
    'feature_extract_epochs': 20,  # Phase 1: frozen backbone
    'fine_tune_epochs': 30,        # Phase 2: end-to-end
    'description': 'ResNet18 with residual connections for efficient training'
}

# Training Hyperparameters (Standardized)
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'total_epochs': 50,
    'optimizer': 'AdamW',
    'weight_decay': 0.01,
    'scheduler': 'StepLR',
    'step_size': 15,
    'gamma': 0.1,
    'device': 'auto'  # Auto-detect best device
}

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'Flowers102',
    'num_classes': 102,
    'total_samples': 8189,
    'train_samples': 1020,
    'val_samples': 1020,
    'test_samples': 6149,
    'image_size': 224,
    'num_workers': 2,
    'pin_memory': True
}

# Data Augmentation Settings
AUGMENTATION_CONFIG = {
    'train': {
        'resize': 256,
        'crop': 224,
        'horizontal_flip': 0.5,
        'rotation': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }
    },
    'val_test': {
        'resize': 224,
        'crop': 224
    }
}

# ImageNet Normalization (Required for pre-trained models)
NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# Progressive Training Strategy
PROGRESSIVE_TRAINING = {
    'phase1': {
        'name': 'Feature Extraction',
        'freeze_backbone': True,
        'epochs': 20,
        'learning_rate': 0.001,
        'description': 'Train classifier head with frozen ResNet18 backbone'
    },
    'phase2': {
        'name': 'End-to-End Fine-tuning',
        'freeze_backbone': False,
        'epochs': 30,
        'learning_rate': 0.0001,  # Lower LR for fine-tuning
        'description': 'Fine-tune entire ResNet18 network'
    }
}

# Expected Performance (ResNet18 benchmarks)
EXPECTED_PERFORMANCE = {
    'phase1_accuracy': 75.0,
    'phase2_accuracy': 85.0,
    'training_time': '15-20 minutes',
    'memory_usage': '~2GB GPU',
    'inference_speed': 'Fast'
}

# Evaluation Metrics
METRICS = {
    'primary': ['accuracy', 'top5_accuracy'],
    'secondary': ['precision', 'recall', 'f1_score'],
    'loss': 'cross_entropy',
    'monitor': 'val_accuracy'
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_interval': 10,
    'save_model': True,
    'save_best_only': True,
    'early_stopping': {
        'patience': 10,
        'min_delta': 0.001
    }
}

# File Paths
PATHS = {
    'data_dir': './data',
    'models_dir': './models',
    'logs_dir': './logs',
    'results_dir': './results'
}

# ResNet18 Architecture Details
ARCHITECTURE_INFO = {
    'layers': 18,
    'blocks': [2, 2, 2, 2],  # BasicBlock repetitions per layer
    'channels': [64, 128, 256, 512],
    'block_type': 'BasicBlock',
    'expansion': 1,
    'features': {
        'conv1': '7x7 conv, 64 channels',
        'layer1': '2x BasicBlock, 64 channels',
        'layer2': '2x BasicBlock, 128 channels', 
        'layer3': '2x BasicBlock, 256 channels',
        'layer4': '2x BasicBlock, 512 channels',
        'avgpool': 'AdaptiveAvgPool2d',
        'fc': 'Linear(512, num_classes)'
    }
}

def get_resnet18_config():
    """Get complete ResNet18 configuration"""
    return {
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'dataset': DATASET_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'normalization': NORMALIZATION,
        'progressive': PROGRESSIVE_TRAINING,
        'expected': EXPECTED_PERFORMANCE,
        'metrics': METRICS,
        'logging': LOGGING_CONFIG,
        'paths': PATHS,
        'architecture': ARCHITECTURE_INFO
    }

def print_resnet18_summary():
    """Print ResNet18 configuration summary"""
    print("🔥 ResNet18 Transfer Learning Configuration")
    print("=" * 60)
    print(f"🏗️  Architecture: {MODEL_CONFIG['architecture'].upper()}")
    print(f"⚙️  Parameters: {MODEL_CONFIG['total_params']}")
    print(f"📊 Dataset: {DATASET_CONFIG['name']} ({DATASET_CONFIG['num_classes']} classes)")
    print(f"🎯 Training: {TRAINING_CONFIG['total_epochs']} epochs ({PROGRESSIVE_TRAINING['phase1']['epochs']} + {PROGRESSIVE_TRAINING['phase2']['epochs']})")
    print(f"📈 Expected: {EXPECTED_PERFORMANCE['phase1_accuracy']}% → {EXPECTED_PERFORMANCE['phase2_accuracy']}%")
    print(f"⏱️  Time: ~{EXPECTED_PERFORMANCE['training_time']}")
    print(f"💾 Memory: {EXPECTED_PERFORMANCE['memory_usage']}")
    print("=" * 60)

if __name__ == "__main__":
    print_resnet18_summary() 