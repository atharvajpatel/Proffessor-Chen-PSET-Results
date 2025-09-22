#!/usr/bin/env python3
"""
CUDA Compatibility Test Script
Tests CUDA availability across different libraries for RTX 4070
"""

import sys
import subprocess
import platform

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_status(library, status, details=""):
    """Print library status with formatting"""
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {library:<20} {details}")

def check_system_info():
    """Check basic system information"""
    print_header("SYSTEM INFORMATION")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check NVIDIA driver
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            print(f"GPU: {gpu_info[0]}")
            print(f"Driver Version: {gpu_info[1]}")
            print(f"VRAM: {gpu_info[2]} MB")
        else:
            print("❌ nvidia-smi not available")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")

def check_pytorch():
    """Check PyTorch CUDA support"""
    print_header("PYTORCH CUDA TEST")
    try:
        import torch
        print_status("PyTorch", True, f"Version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print_status("CUDA Available", cuda_available)
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print_status("GPU Count", True, f"{device_count} device(s)")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print_status(f"GPU {i}", True, f"{gpu_name} ({memory:.1f} GB)")
            
            # Test tensor operations
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                print_status("GPU Tensor Ops", True, "Matrix multiplication successful")
            except Exception as e:
                print_status("GPU Tensor Ops", False, f"Error: {e}")
        else:
            print_status("CUDA Support", False, "PyTorch built without CUDA")
            
    except ImportError:
        print_status("PyTorch", False, "Not installed")
    except Exception as e:
        print_status("PyTorch", False, f"Error: {e}")

def check_xgboost():
    """Check XGBoost GPU support"""
    print_header("XGBOOST GPU TEST")
    try:
        import xgboost as xgb
        print_status("XGBoost", True, f"Version: {xgb.__version__}")
        
        # Test GPU training
        try:
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            
            # Try GPU training
            model = xgb.XGBClassifier(
                tree_method='hist',
                device='cuda',
                n_estimators=10,
                verbosity=0
            )
            model.fit(X, y)
            print_status("GPU Training", True, "XGBoost GPU training successful")
            
        except Exception as e:
            print_status("GPU Training", False, f"Error: {e}")
            
    except ImportError:
        print_status("XGBoost", False, "Not installed")
    except Exception as e:
        print_status("XGBoost", False, f"Error: {e}")

def check_cupy():
    """Check CuPy GPU arrays"""
    print_header("CUPY GPU ARRAYS TEST")
    try:
        import cupy as cp
        print_status("CuPy", True, f"Version: {cp.__version__}")
        
        # Test GPU array operations
        try:
            x_gpu = cp.random.randn(1000, 1000)
            y_gpu = cp.random.randn(1000, 1000)
            z_gpu = cp.dot(x_gpu, y_gpu)
            print_status("GPU Arrays", True, "CuPy array operations successful")
            
            # Memory info
            mempool = cp.get_default_memory_pool()
            print_status("GPU Memory", True, f"Used: {mempool.used_bytes() / (1024**2):.1f} MB")
            
        except Exception as e:
            print_status("GPU Arrays", False, f"Error: {e}")
            
    except ImportError:
        print_status("CuPy", False, "Not installed")
    except Exception as e:
        print_status("CuPy", False, f"Error: {e}")

def check_sklearn():
    """Check scikit-learn (CPU baseline)"""
    print_header("SCIKIT-LEARN (CPU BASELINE)")
    try:
        import sklearn
        print_status("Scikit-learn", True, f"Version: {sklearn.__version__}")
        
        # Quick performance test
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        import time
        
        X, y = make_classification(n_samples=10000, n_features=100, random_state=42)
        
        start_time = time.time()
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        cpu_time = time.time() - start_time
        
        print_status("CPU Training", True, f"LogReg trained in {cpu_time:.3f}s")
        
    except ImportError:
        print_status("Scikit-learn", False, "Not installed")
    except Exception as e:
        print_status("Scikit-learn", False, f"Error: {e}")

def performance_comparison():
    """Compare CPU vs GPU performance where possible"""
    print_header("PERFORMANCE COMPARISON")
    
    try:
        import torch
        import time
        import numpy as np
        
        # Matrix multiplication benchmark
        size = 2000
        
        # CPU test
        start = time.time()
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        # GPU test (if available)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure GPU is ready
            start = time.time()
            x_gpu = torch.randn(size, size).cuda()
            y_gpu = torch.randn(size, size).cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time
            print_status("CPU MatMul", True, f"{cpu_time:.3f}s")
            print_status("GPU MatMul", True, f"{gpu_time:.3f}s")
            print_status("GPU Speedup", True, f"{speedup:.1f}x faster")
        else:
            print_status("GPU Comparison", False, "CUDA not available")
            
    except Exception as e:
        print_status("Performance Test", False, f"Error: {e}")

def main():
    """Run all CUDA compatibility tests"""
    print_header("CUDA COMPATIBILITY CHECK FOR RTX 4070")
    print("Testing CUDA support across ML libraries...")
    
    check_system_info()
    check_pytorch()
    check_xgboost()
    check_cupy()
    check_sklearn()
    performance_comparison()
    
    print_header("SUMMARY")
    print("✅ = Working correctly")
    print("❌ = Not working or not installed")
    print("\nFor full GPU acceleration, ensure PyTorch and XGBoost show ✅")
    print("CuPy is optional but provides additional GPU array operations")
    print("\nYour RTX 4070 is ready for GPU-accelerated machine learning!")

if __name__ == "__main__":
    main()
