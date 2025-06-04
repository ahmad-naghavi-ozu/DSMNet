#!/usr/bin/env python3

"""
Test script to validate DSMNet PyTorch implementation
Run this after installing PyTorch dependencies with install_pytorch.sh
"""

import sys
import os

def test_imports():
    """Test if all PyTorch modules can be imported"""
    print("Testing PyTorch imports...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch not installed. Run install_pytorch.sh first.")
        return False
    
    try:
        import config
        print("✓ config.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing config: {e}")
        return False
    
    try:
        from nets import MTLNet, UNet, DenoiseNet
        print("✓ PyTorch neural networks imported successfully")
    except ImportError as e:
        print(f"✗ Error importing nets: {e}")
        return False
    
    try:
        from utils import DSMDataset, create_data_loaders
        print("✓ PyTorch utilities imported successfully")
    except ImportError as e:
        print(f"✗ Error importing utils: {e}")
        return False
    
    try:
        from metrics import HeightMetrics, SegmentationMetrics
        print("✓ PyTorch metrics imported successfully")
    except ImportError as e:
        print(f"✗ Error importing metrics: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    
    try:
        import torch
        from nets import MTLNet, DenoiseNet
        
        # Test MTL model creation
        mtl_model = MTLNet(num_classes=6)
        print("✓ MTL model created successfully")
        
        # Test DAE model creation  
        dae_model = DenoiseNet()
        print("✓ DAE model created successfully")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            mtl_outputs = mtl_model(dummy_input)
            print("✓ MTL forward pass successful")
            
            # Create dummy MTL outputs for DAE
            dae_input = torch.cat([dummy_input, mtl_outputs[0], mtl_outputs[1], mtl_outputs[2]], dim=1)
            dae_output = dae_model(dae_input)
            print("✓ DAE forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model creation: {e}")
        return False

def test_python_version():
    """Test if Python 3.11 is being used"""
    print("\nTesting Python version...")
    
    try:
        import sys
        python_version = sys.version_info
        version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        print(f"✓ Python version: {version_str}")
        
        if python_version.major == 3 and python_version.minor == 11:
            print("✓ Python 3.11 detected - Perfect!")
        elif python_version.major == 3 and python_version.minor >= 8:
            print(f"⚠️  Python {version_str} detected - Compatible but Python 3.11 recommended")
        else:
            print(f"✗ Python {version_str} detected - Python 3.8+ required, 3.11 recommended")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking Python version: {e}")
        return False

def test_config():
    """Test configuration values"""
    print("\nTesting configuration...")
    
    try:
        import config
        
        # Check if PyTorch device is configured
        if hasattr(config, 'DEVICE'):
            print(f"✓ Device configured: {config.DEVICE}")
        
        if hasattr(config, 'USE_MULTI_GPU'):
            print(f"✓ Multi-GPU setting: {config.USE_MULTI_GPU}")
        
        if hasattr(config, 'GPU_IDS'):
            print(f"✓ GPU IDs: {config.GPU_IDS}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing config: {e}")
        return False

def main():
    """Main test function"""
    print("=== DSMNet PyTorch Implementation Test ===")
    print("Testing Python 3.11 compatibility and PyTorch setup")
    print()
    
    # Test Python version
    if not test_python_version():
        print("\n✗ Python version tests failed")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed")
        print("\nTo fix this:")
        print("1. Run: ./install_py311.sh")
        print("2. Activate environment: conda activate dsmnet_pytorch_py311")
        print("3. Run this test again: python test_setup.py")
        sys.exit(1)
    
    # Test configuration
    if not test_config():
        print("\n✗ Configuration tests failed")
        sys.exit(1)
    
    # Test model creation
    if not test_model_creation():
        print("\n✗ Model creation tests failed")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✅ All tests passed!")
    print("🎉 DSMNet PyTorch implementation is ready to use!")
    print("="*50)
    
    print("\n📋 Next steps:")
    print("1. 📁 Prepare your dataset in the appropriate directory structure")
    print("2. ⚙️  Update config.py with your dataset paths and parameters")
    print("3. 🚀 Run training: ./run_train_mtl.sh")
    print("4. 🧪 Run testing: ./run_test.sh")
    print("5. 🔄 Run full pipeline: ./run_full_pipeline.sh")
    
    print("\n💡 Quick tips:")
    print("   • Use 'conda activate dsmnet_pytorch_py311' to activate environment")
    print("   • Check GPU availability with 'nvidia-smi'")

if __name__ == "__main__":
    main()
