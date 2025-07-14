#!/usr/bin/env python3
"""
Basic usage example for FANN Rust Core neural networks.

This example demonstrates:
- Creating a simple neural network
- Training on synthetic data
- Making predictions
- Evaluating performance
- Saving and loading models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time

try:
    import fann_rust_core as fann
except ImportError:
    print("fann_rust_core not installed. Please install with: pip install fann-rust-core")
    exit(1)

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data."""
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create a simple pattern: sum of first half features > sum of second half
    first_half = X[:, :n_features//2].sum(axis=1)
    second_half = X[:, n_features//2:].sum(axis=1)
    
    # Binary classification
    y_binary = (first_half > second_half).astype(int)
    
    # Convert to one-hot encoding
    y = np.zeros((n_samples, 2), dtype=np.float32)
    y[np.arange(n_samples), y_binary] = 1.0
    
    return X, y

def demonstrate_basic_network():
    """Demonstrate basic neural network creation and usage."""
    print("=== Basic Neural Network Demo ===")
    
    # Check library capabilities
    print(f"Library version: {fann.__version__}")
    env_info = fann.check_environment()
    print(f"SIMD features: {env_info.get('simd_features', 'None')}")
    print(f"Thread count: {env_info.get('thread_count', 'Unknown')}")
    print()
    
    # Generate data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=20)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Create neural network
    print("Creating neural network...")
    network = fann.create_network(
        layers=[20, 64, 32, 2],  # Input(20) -> Hidden(64) -> Hidden(32) -> Output(2)
        activation="relu",
        output_activation="softmax"
    )
    
    print(f"Network created: {network}")
    print(f"Input size: {network.input_size}")
    print(f"Output size: {network.output_size}")
    print(f"Total parameters: {network.parameter_count}")
    print(f"Memory footprint: {network.memory_footprint / 1024:.2f} KB")
    print()
    
    # Train the network
    print("Training network...")
    start_time = time.time()
    
    results = network.train(
        inputs=X_train,
        targets=y_train,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        validation_split=0.2,
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final loss: {results.final_loss:.6f}")
    print(f"Final accuracy: {results.final_accuracy:.4f}")
    print(f"Epochs completed: {results.epochs_completed}")
    print(f"Converged: {results.converged}")
    print()
    
    # Make predictions
    print("Making predictions on test set...")
    start_time = time.time()
    predictions = network.predict(X_test)
    prediction_time = time.time() - start_time
    
    print(f"Prediction completed in {prediction_time:.4f} seconds")
    print(f"Prediction shape: {predictions.shape}")
    print()
    
    # Evaluate performance
    print("Evaluating performance...")
    eval_results = network.evaluate(X_test, y_test)
    
    print(f"Test loss: {eval_results['loss']:.6f}")
    print(f"Test accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test samples: {eval_results['samples']}")
    print()
    
    # Calculate additional metrics
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Manual accuracy calculation: {accuracy:.4f}")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        pred_class = predicted_classes[i]
        true_class = true_classes[i]
        confidence = predictions[i, pred_class]
        print(f"Sample {i}: Predicted={pred_class}, True={true_class}, Confidence={confidence:.4f}")
    
    print()
    
    # Save the model
    model_path = "demo_model.json"
    print(f"Saving model to {model_path}...")
    network.save(model_path)
    
    # Load the model
    print("Loading model...")
    loaded_network = fann.load_network(model_path)
    
    # Verify loaded model works
    loaded_predictions = loaded_network.predict(X_test[:1])
    original_predictions = network.predict(X_test[:1])
    
    prediction_diff = np.abs(loaded_predictions - original_predictions).max()
    print(f"Max prediction difference after loading: {prediction_diff:.8f}")
    
    if prediction_diff < 1e-6:
        print("✓ Model saved and loaded successfully!")
    else:
        print("⚠ Model loading may have issues")
    
    return network, results

def demonstrate_training_visualization(network, results):
    """Visualize training progress."""
    print("\n=== Training Visualization ===")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        epochs = range(1, len(results.training_loss_history) + 1)
        ax1.plot(epochs, results.training_loss_history, label='Training Loss', color='blue')
        if results.validation_loss_history:
            ax1.plot(epochs, results.validation_loss_history, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, results.training_accuracy_history, label='Training Accuracy', color='blue')
        if results.validation_accuracy_history:
            ax2.plot(epochs, results.validation_accuracy_history, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Progress - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print("Training progress plot saved to 'training_progress.png'")
        
        # Show if in interactive mode
        try:
            plt.show()
        except:
            pass  # Non-interactive environment
            
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Visualization error: {e}")

def demonstrate_performance_benchmarks():
    """Demonstrate performance benchmarking capabilities."""
    print("\n=== Performance Benchmarks ===")
    
    try:
        # Matrix multiplication benchmark
        print("Running matrix multiplication benchmark...")
        matrix_results = fann.benchmarks.benchmark_matrix_multiply(size=256, iterations=10)
        
        for operation, metrics in matrix_results.items():
            if isinstance(metrics, dict) and 'duration_ms' in metrics:
                duration = metrics['duration_ms']
                throughput = metrics.get('throughput', 'N/A')
                print(f"  {operation}: {duration}ms (throughput: {throughput})")
        
        print()
        
        # Activation function benchmarks
        print("Running activation function benchmarks...")
        activation_results = fann.benchmarks.benchmark_activations(size=10000, iterations=100)
        
        for activation, metrics in activation_results.items():
            if isinstance(metrics, dict):
                scalar_time = metrics.get('scalar_time', 0)
                simd_time = metrics.get('simd_time', 0)
                speedup = metrics.get('speedup', 1.0)
                print(f"  {activation}: {speedup:.2f}x speedup (SIMD vs scalar)")
        
        print()
        
    except Exception as e:
        print(f"Benchmark error: {e}")

def demonstrate_utilities():
    """Demonstrate utility functions."""
    print("\n=== Utility Functions ===")
    
    # Test data
    test_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    try:
        # Softmax
        softmax_result = fann.utils.softmax(test_vector)
        print(f"Softmax of {test_vector}: {softmax_result}")
        print(f"Softmax sum: {softmax_result.sum():.6f} (should be ~1.0)")
        
        # One-hot encoding
        class_index = 2
        num_classes = 5
        one_hot = fann.utils.one_hot_encode(class_index, num_classes)
        print(f"One-hot encoding of class {class_index}: {one_hot}")
        
        # One-hot decoding
        decoded_class = fann.utils.one_hot_decode(one_hot)
        print(f"Decoded class: {decoded_class} (should be {class_index})")
        
        # Matrix multiplication
        A = np.random.rand(3, 4).astype(np.float32)
        B = np.random.rand(4, 2).astype(np.float32)
        C = fann.utils.matrix_multiply(A, B)
        
        print(f"\nMatrix multiplication:")
        print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
        
        # Verify with NumPy
        C_numpy = A @ B
        max_diff = np.abs(C - C_numpy).max()
        print(f"Max difference from NumPy result: {max_diff:.8f}")
        
    except Exception as e:
        print(f"Utility function error: {e}")

def main():
    """Main demonstration function."""
    print("FANN Rust Core - Comprehensive Usage Example")
    print("=" * 50)
    
    try:
        # Basic network demonstration
        network, results = demonstrate_basic_network()
        
        # Training visualization
        demonstrate_training_visualization(network, results)
        
        # Performance benchmarks
        demonstrate_performance_benchmarks()
        
        # Utility functions
        demonstrate_utilities()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Check the generated files:")
        print("  - demo_model.json: Saved neural network model")
        print("  - training_progress.png: Training visualization (if matplotlib available)")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()