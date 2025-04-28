import torch
import time
import numpy as np
import gc
import psutil
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def print_memory_stats(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"{label} GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # System memory
    process = psutil.Process(os.getpid())
    system_memory = process.memory_info().rss / (1024 ** 3)
    print(f"{label} System Memory: {system_memory:.2f} GB")
    print("-" * 50)

class DeepRLModel(torch.nn.Module):
    """A large neural network for RL tasks."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8):
        super(DeepRLModel, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

def matrix_operations_test(matrix_size, iterations, device):
    """Test large matrix operations."""
    print(f"Running matrix operations test with {matrix_size}x{matrix_size} matrices...")
    
    operation_times = []
    
    for i in range(iterations):
        # Create large matrices
        matrix_a = torch.rand(matrix_size, matrix_size, device=device)
        matrix_b = torch.rand(matrix_size, matrix_size, device=device)
        
        # Synchronize before timing
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        # Matrix multiplication
        result = torch.matmul(matrix_a, matrix_b)
        
        # Element-wise operations
        result = torch.sin(result) + torch.cos(matrix_a)
        result = torch.relu(result)
        
        # Reduction operations
        _ = torch.mean(result, dim=1)
        _ = torch.max(result, dim=0)[0]
        
        # Ensure operations are completed
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        operation_times.append(end_time - start_time)
        
        print(f"  Iteration {i+1}/{iterations}: {operation_times[-1]:.4f} seconds")
    
    avg_time = sum(operation_times) / len(operation_times)
    print(f"Average matrix operation time: {avg_time:.4f} seconds\n")
    return avg_time

def memory_allocation_test(target_gb, device):
    """Gradually allocate memory up to target_gb."""
    print(f"Running memory allocation test (target: {target_gb} GB)...")
    
    # Start with 1GB chunks
    chunk_size_gb = 1
    tensors = []
    current_allocation = 0
    allocation_times = []
    
    while current_allocation < target_gb:
        # Calculate elements needed for this chunk
        elements = int(chunk_size_gb * (1024**3) / 4)  # 4 bytes per float32
        
        try:
            start_time = time.time()
            tensor = torch.rand(elements, dtype=torch.float32, device=device)
            # Perform some operations to ensure it's used
            tensor = tensor * 2 + 1
            tensors.append(tensor)
            
            end_time = time.time()
            allocation_times.append(end_time - start_time)
            
            current_allocation += chunk_size_gb
            print(f"  Allocated {chunk_size_gb} GB chunk in {allocation_times[-1]:.4f} seconds (Total: {current_allocation} GB)")
            
            print_memory_stats("  Current")
        except RuntimeError as e:
            if "out of memory" in str(e):
                if chunk_size_gb <= 0.1:
                    print(f"  Cannot allocate more memory. Reached {current_allocation:.2f} GB")
                    break
                # Try with smaller chunk
                chunk_size_gb /= 2
                print(f"  Memory allocation failed. Trying with {chunk_size_gb} GB chunks")
            else:
                print(f"  Error: {e}")
                break
    
    if tensors:
        avg_time = sum(allocation_times) / len(allocation_times)
        print(f"Average allocation time per chunk: {avg_time:.4f} seconds")
        print(f"Total allocated: {current_allocation:.2f} GB\n")
    
    return current_allocation, tensors

def training_throughput_test(batch_size, model_size, num_batches, device):
    """Test training throughput with a large model."""
    print(f"Running training throughput test with batch size {batch_size}...")
    
    # Create a large model
    input_dim = 1024
    hidden_dim = model_size
    output_dim = 512
    model = DeepRLModel(input_dim, hidden_dim, output_dim).to(device)
    
    # Print model size
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    # Create synthetic data
    data = torch.rand(batch_size * num_batches, input_dim, device=device)
    targets = torch.rand(batch_size * num_batches, output_dim, device=device)
    
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    # Track metrics
    batch_times = []
    losses = []
    
    # Training loop
    model.train()
    start_total = time.time()
    
    for i, (inputs, targets) in enumerate(dataloader):
        batch_start = time.time()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record time and loss
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        losses.append(loss.item())
        
        if (i+1) % 5 == 0:
            print(f"  Batch {i+1}/{num_batches}: Time = {batch_time:.4f}s, Loss = {loss.item():.6f}")
        
        # Optional: Force GPU synchronization to get accurate timings
        torch.cuda.synchronize() if device.type == "cuda" else None
    
    end_total = time.time()
    total_time = end_total - start_total
    
    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    samples_per_second = batch_size / avg_batch_time
    
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Average batch time: {avg_batch_time:.4f} seconds")
    print(f"Throughput: {samples_per_second:.2f} samples/second\n")
    
    return avg_batch_time, samples_per_second, losses

def inference_benchmark(model, input_size, num_inferences, batch_size, device):
    """Test inference speed."""
    print(f"Running inference benchmark with batch size {batch_size}...")
    
    # Generate test data
    test_data = torch.rand(num_inferences, input_size, device=device)
    
    # Put model in evaluation mode
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_data[:batch_size])
    
    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, num_inferences, batch_size):
            end_idx = min(i + batch_size, num_inferences)
            _ = model(test_data[i:end_idx])
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    
    total_time = end_time - start_time
    inferences_per_second = num_inferences / total_time
    
    print(f"Performed {num_inferences} inferences in {total_time:.4f} seconds")
    print(f"Inference throughput: {inferences_per_second:.2f} inferences/second\n")
    
    return inferences_per_second

def plot_performance_results(results):
    """Plot the performance results."""
    metrics = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(12, 8))
    plt.bar(metrics, values, color='royalblue')
    plt.title('GPU Performance Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('gpu_performance_results.png')
    print("Performance results plotted and saved as 'gpu_performance_results.png'")

def comprehensive_gpu_test():
    """Run a comprehensive GPU test allocating ~10GB memory."""
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    
    print_memory_stats("Initial")
    
    results = {}
    
    # 1. Memory Allocation Test - Try to allocate 10GB
    target_memory = 10  # GB
    allocated_gb, tensors = memory_allocation_test(target_memory, device)
    results["Memory Allocated (GB)"] = allocated_gb
    
    print_memory_stats("After memory allocation")
    
    # 2. Large Matrix Operations
    matrix_size = 10000  # 10k x 10k matrices
    iterations = 5
    avg_matrix_time = matrix_operations_test(matrix_size, iterations, device)
    results["Matrix Ops Time (s)"] = avg_matrix_time
    
    print_memory_stats("After matrix operations")
    
    # 3. Training Throughput Test with a large model
    batch_size = 64
    model_size = 4096  # hidden dimensions
    num_batches = 20
    avg_batch_time, samples_per_second, losses = training_throughput_test(
        batch_size, model_size, num_batches, device
    )
    results["Training Throughput (samples/s)"] = samples_per_second
    results["Average Batch Time (s)"] = avg_batch_time
    
    print_memory_stats("After training test")
    
    # 4. Inference Benchmark
    input_size = 1024
    num_inferences = 1000
    inference_batch_size = 128
    
    # Re-use model from training test
    try:
        model = DeepRLModel(input_size, model_size, 512).to(device)
        inferences_per_second = inference_benchmark(
            model, input_size, num_inferences, inference_batch_size, device
        )
        results["Inference Throughput (infs/s)"] = inferences_per_second
    except Exception as e:
        print(f"Error during inference benchmark: {e}")
    
    print_memory_stats("After inference benchmark")
    
    # 5. Multi-GPU test (if available)
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs. Running multi-GPU test...")
        # Add multi-GPU test here
    
    # Calculate a performance score
    if device.type == "cuda":
        # Normalize and combine metrics (higher is better)
        performance_score = (
            allocated_gb * 10 +  # Memory capacity
            (1/avg_matrix_time) * 1000 +  # Matrix operations speed
            samples_per_second / 10 +  # Training throughput
            inferences_per_second / 100  # Inference throughput
        )
        results["Performance Score"] = performance_score
        
        # Add a rating
        if performance_score > 2000:
            rating = "Exceptional - Top-tier GPU performance"
        elif performance_score > 1500:
            rating = "Excellent - High-end GPU performance"
        elif performance_score > 1000:
            rating = "Very Good - Capable of handling demanding RL tasks"
        elif performance_score > 500:
            rating = "Good - Suitable for most RL applications"
        else:
            rating = "Moderate - May struggle with the most demanding RL models"
        
        print(f"\n===== GPU Performance Results =====")
        print(f"Memory Allocated: {allocated_gb:.2f} GB")
        print(f"Matrix Operations Time (10kx10k): {avg_matrix_time:.4f} seconds")
        print(f"Training Throughput: {samples_per_second:.2f} samples/second")
        print(f"Inference Throughput: {inferences_per_second:.2f} inferences/second")
        print(f"\nPerformance Score: {performance_score:.2f}")
        print(f"Rating: {rating}")
        
        # Plot results
        try:
            plot_performance_results(results)
        except Exception as e:
            print(f"Could not plot results: {e}")
    
    # Clean up
    print("\nCleaning up...")
    del tensors
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    gc.collect()
    print_memory_stats("Final")

if __name__ == "__main__":
    comprehensive_gpu_test()