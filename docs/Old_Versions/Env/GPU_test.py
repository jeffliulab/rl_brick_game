import torch
import sys

def test_gpu():
    # Print Python version
    print(f"Python version: {sys.version}")
    
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Print CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Get the name of the current GPU
        current_gpu = torch.cuda.current_device()
        print(f"Current GPU: {current_gpu}")
        print(f"GPU name: {torch.cuda.get_device_name(current_gpu)}")
        
        # Test a simple tensor operation on GPU
        print("\nRunning a simple GPU test...")
        
        # Create a tensor on GPU
        x = torch.ones(1000, 1000, device='cuda')
        y = torch.ones(1000, 1000, device='cuda')
        
        # Perform a matrix multiplication
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        # Synchronize CUDA
        torch.cuda.synchronize()
        
        # Calculate elapsed time
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"GPU operation completed. Result shape: {z.shape}")
        print(f"Time taken: {elapsed_time:.2f} ms")
        
        # Memory usage
        print(f"\nCUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("CUDA is not available. GPU acceleration cannot be used.")

if __name__ == "__main__":
    test_gpu()