#!/usr/bin/env python3
"""
MNIST Classification with FANN-Rust-Core

This example demonstrates advanced neural network training on the MNIST dataset
using FANN-Rust-Core Python bindings with optimization features.
"""

import fann_rust_core as fann
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import time
import os

def main():
    print("FANN-Rust-Core: MNIST Classification Example")
    print("============================================")
    
    # Load MNIST dataset
    print("📁 Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    X_train, X_test = preprocess_data(X_train, X_test)
    y_train_onehot, y_test_onehot = prepare_labels(y_train, y_test)
    
    # Create optimized network
    print("\n🏗️  Creating optimized neural network...")
    network = create_optimized_network()
    
    # Train with monitoring
    print("\n🚀 Training with performance monitoring...")
    training_history = train_with_monitoring(network, X_train, y_train_onehot, X_test, y_test_onehot)
    
    # Comprehensive evaluation
    print("\n📊 Comprehensive evaluation...")
    evaluate_comprehensive(network, X_test, y_test, y_test_onehot)
    
    # Demonstrate optimization features
    print("\n⚡ Optimization demonstrations...")
    demonstrate_optimizations(network, X_train[:1000], X_test[:100])
    
    # Visualizations
    print("\n📈 Creating visualizations...")
    create_visualizations(training_history, network, X_test, y_test)
    
    print("\n✅ MNIST classification example completed!")

def load_mnist_data():
    """Load MNIST dataset (mock implementation for example)."""
    try:
        # Try to use tensorflow/keras datasets if available
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("✓ Loaded real MNIST dataset via TensorFlow")
        return (X_train, y_train), (X_test, y_test)
    except ImportError:
        print("ℹ️  TensorFlow not available, generating mock MNIST data...")
        return generate_mock_mnist()

def generate_mock_mnist():
    """Generate mock MNIST-like data for demonstration."""
    np.random.seed(42)
    
    # Generate training data
    X_train = np.random.randint(0, 256, (5000, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 5000)
    
    # Generate test data
    X_test = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, 1000)
    
    print("✓ Generated mock MNIST dataset")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    return (X_train, y_train), (X_test, y_test)

def preprocess_data(X_train, X_test):
    """Preprocess image data."""
    # Flatten images from 28x28 to 784
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    
    # Normalize to [0, 1]
    X_train_flat = X_train_flat / 255.0
    X_test_flat = X_test_flat / 255.0
    
    # Optional: Additional normalization (zero mean, unit variance)
    # mean = X_train_flat.mean(axis=0)
    # std = X_train_flat.std(axis=0) + 1e-8
    # X_train_flat = (X_train_flat - mean) / std
    # X_test_flat = (X_test_flat - mean) / std
    
    print(f"✓ Preprocessed data:")
    print(f"   Shape: {X_train_flat.shape[0]} x {X_train_flat.shape[1]}")
    print(f"   Range: [{X_train_flat.min():.3f}, {X_train_flat.max():.3f}]")
    
    return X_train_flat, X_test_flat

def prepare_labels(y_train, y_test):
    """Convert labels to one-hot encoding."""
    num_classes = 10
    
    # One-hot encode labels
    y_train_onehot = np.eye(num_classes)[y_train].astype(np.float32)
    y_test_onehot = np.eye(num_classes)[y_test].astype(np.float32)
    
    print(f"✓ Labels converted to one-hot encoding ({num_classes} classes)")
    
    return y_train_onehot, y_test_onehot

def create_optimized_network():
    """Create an optimized neural network for MNIST."""
    network = fann.NeuralNetwork(
        layers=[784, 512, 256, 128, 10],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.001,
        optimizer='adam',
        use_simd=True,
        quantization=None,  # We'll demonstrate quantization separately
        dropout_rates=[0.0, 0.3, 0.4, 0.5, 0.0],  # Dropout for hidden layers
        batch_normalization=[False, True, True, True, False]
    )
    
    print("✓ Created optimized network:")
    print("   Architecture: 784 -> 512 -> 256 -> 128 -> 10")
    print("   Activation: ReLU -> Softmax")
    print("   Optimizer: Adam")
    print("   Features: SIMD, Dropout, Batch Normalization")
    
    return network

def train_with_monitoring(network, X_train, y_train, X_val, y_val):
    """Train network with comprehensive monitoring."""
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
        'training_time': []
    }
    
    def training_callback(epoch, train_loss, train_acc, val_loss=None, val_acc=None, lr=None):
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_acc)
        training_history['val_loss'].append(val_loss or 0.0)
        training_history['val_accuracy'].append(val_acc or 0.0)
        training_history['learning_rate'].append(lr or 0.001)
        training_history['training_time'].append(time.time())
        
        if epoch % 5 == 0:
            val_str = f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}" if val_loss else ""
            print(f"   Epoch {epoch:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}{val_str}")
    
    print("   Starting training with early stopping and learning rate scheduling...")
    
    start_time = time.time()
    
    try:
        final_error = network.train(
            inputs=X_train.tolist(),
            targets=y_train.tolist(),
            epochs=100,
            batch_size=128,
            validation_data=(X_val.tolist(), y_val.tolist()),
            early_stopping=True,
            patience=10,
            min_delta=0.001,
            learning_rate_schedule='plateau',
            lr_patience=5,
            lr_factor=0.5,
            callback=training_callback
        )
    except Exception as e:
        # Fallback to basic training if advanced features aren't available
        print(f"   Advanced training failed ({e}), using basic training...")
        final_error = network.train(
            inputs=X_train.tolist(),
            targets=y_train.tolist(),
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            callback=training_callback
        )
    
    training_time = time.time() - start_time
    
    print(f"✅ Training completed!")
    print(f"   Final training error: {final_error:.6f}")
    print(f"   Total training time: {training_time:.1f}s")
    print(f"   Epochs trained: {len(training_history['epoch'])}")
    
    return training_history

def evaluate_comprehensive(network, X_test, y_test, y_test_onehot):
    """Comprehensive evaluation of the trained network."""
    print("   Running inference on test set...")
    
    # Measure inference time
    start_time = time.time()
    predictions = network.predict_batch(X_test.tolist())
    inference_time = time.time() - start_time
    
    # Convert predictions to class labels
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Test Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Inference time: {inference_time:.3f}s ({inference_time/len(X_test)*1000:.2f}ms per sample)")
    print(f"   Throughput: {len(X_test)/inference_time:.0f} samples/second")
    
    # Detailed classification report
    print("\n📊 Per-class Performance:")
    class_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # Top errors analysis
    print("\n🔍 Error Analysis:")
    analyze_errors(y_test, y_pred, predictions)
    
    return accuracy

def analyze_errors(y_true, y_pred, predictions):
    """Analyze prediction errors."""
    errors = y_true != y_pred
    error_count = errors.sum()
    total_count = len(y_true)
    
    print(f"   Total errors: {error_count}/{total_count} ({error_count/total_count*100:.2f}%)")
    
    if error_count > 0:
        # Find most common error types
        error_indices = np.where(errors)[0]
        error_pairs = [(y_true[i], y_pred[i]) for i in error_indices]
        
        from collections import Counter
        common_errors = Counter(error_pairs).most_common(5)
        
        print("   Most common error types:")
        for (true_class, pred_class), count in common_errors:
            percentage = count / error_count * 100
            print(f"     {true_class} -> {pred_class}: {count} times ({percentage:.1f}% of errors)")
        
        # Find least confident correct predictions
        correct_indices = np.where(~errors)[0]
        if len(correct_indices) > 0:
            correct_confidences = [np.max(predictions[i]) for i in correct_indices]
            least_confident_idx = correct_indices[np.argmin(correct_confidences)]
            min_confidence = np.min(correct_confidences)
            print(f"   Least confident correct prediction: class {y_true[least_confident_idx]} with {min_confidence:.3f} confidence")

def demonstrate_optimizations(network, train_sample, test_sample):
    """Demonstrate various optimization features."""
    print("🔧 SIMD Optimization:")
    demonstrate_simd_performance(network, test_sample)
    
    print("\n⚡ Model Quantization:")
    demonstrate_quantization(network, train_sample, test_sample)
    
    print("\n📊 Memory Optimization:")
    demonstrate_memory_optimization(network)

def demonstrate_simd_performance(network, test_data):
    """Demonstrate SIMD performance benefits."""
    # Benchmark with SIMD enabled
    start_time = time.time()
    for _ in range(100):
        _ = network.predict_batch([row.tolist() for row in test_data])
    simd_time = time.time() - start_time
    
    try:
        # Create network without SIMD for comparison
        network_no_simd = fann.NeuralNetwork(
            layers=[784, 512, 256, 128, 10],
            use_simd=False
        )
        # Copy weights from trained network
        network_no_simd.set_weights(network.get_weights())
        
        # Benchmark without SIMD
        start_time = time.time()
        for _ in range(100):
            _ = network_no_simd.predict_batch([row.tolist() for row in test_data])
        no_simd_time = time.time() - start_time
        
        speedup = no_simd_time / simd_time
        print(f"   SIMD enabled: {simd_time:.3f}s")
        print(f"   SIMD disabled: {no_simd_time:.3f}s")
        print(f"   SIMD speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"   SIMD comparison failed: {e}")
        print(f"   SIMD benchmark time: {simd_time:.3f}s")

def demonstrate_quantization(network, calibration_data, test_data):
    """Demonstrate model quantization."""
    try:
        from fann_rust_core import QuantizationEngine
        
        # Create quantization engine
        quantizer = QuantizationEngine()
        
        # Quantize the model
        print("   Calibrating quantization parameters...")
        quantized_network = quantizer.quantize(
            network=network,
            calibration_data=calibration_data[:100].tolist(),
            quantization_type='int8'
        )
        
        # Compare model sizes
        original_size = network.get_model_size()
        quantized_size = quantized_network.get_model_size()
        compression_ratio = original_size / quantized_size
        
        print(f"   Original model size: {original_size/1024:.1f}KB")
        print(f"   Quantized model size: {quantized_size/1024:.1f}KB")
        print(f"   Compression ratio: {compression_ratio:.1f}x")
        
        # Compare inference speeds
        start_time = time.time()
        original_outputs = network.predict_batch([row.tolist() for row in test_data])
        original_time = time.time() - start_time
        
        start_time = time.time()
        quantized_outputs = quantized_network.predict_batch([row.tolist() for row in test_data])
        quantized_time = time.time() - start_time
        
        # Compare accuracy
        accuracy_diff = np.mean([
            np.abs(np.array(orig) - np.array(quant)).mean()
            for orig, quant in zip(original_outputs[:10], quantized_outputs[:10])
        ])
        
        print(f"   Original inference time: {original_time:.3f}s")
        print(f"   Quantized inference time: {quantized_time:.3f}s")
        print(f"   Speedup: {original_time/quantized_time:.2f}x")
        print(f"   Average output difference: {accuracy_diff:.6f}")
        
    except Exception as e:
        print(f"   Quantization demonstration failed: {e}")

def demonstrate_memory_optimization(network):
    """Demonstrate memory optimization features."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Get network memory usage
        network_memory = network.get_memory_usage()
        
        print(f"   Process memory usage: {memory_before:.1f}MB")
        print(f"   Network memory usage: {network_memory/1024/1024:.1f}MB")
        print(f"   Memory efficiency: {network_memory/(memory_before*1024*1024)*100:.1f}% of total")
        
    except Exception as e:
        print(f"   Memory monitoring failed: {e}")

def create_visualizations(training_history, network, X_test, y_test):
    """Create training and evaluation visualizations."""
    try:
        # Training curves
        plot_training_curves(training_history)
        
        # Confusion matrix
        plot_confusion_matrix(network, X_test, y_test)
        
        # Sample predictions
        plot_sample_predictions(network, X_test, y_test)
        
        print("✓ Visualizations saved to disk")
        
    except ImportError:
        print("📊 Matplotlib/Seaborn not available - skipping visualizations")
    except Exception as e:
        print(f"📊 Visualization failed: {e}")

def plot_training_curves(history):
    """Plot training loss and accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # Training loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if any(history['val_loss']):
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    if any(history['val_accuracy']):
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    if any(history['learning_rate']):
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Training time per epoch
    if len(history['training_time']) > 1:
        time_diffs = np.diff(history['training_time'])
        axes[1, 1].plot(epochs[1:], time_diffs, 'purple')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('mnist_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(network, X_test, y_test):
    """Plot confusion matrix."""
    predictions = network.predict_batch(X_test.tolist())
    y_pred = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('MNIST Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('mnist_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_sample_predictions(network, X_test, y_test):
    """Plot sample predictions with confidence."""
    # Reshape for visualization (assuming flattened MNIST)
    X_test_images = X_test.reshape(-1, 28, 28)
    
    # Get predictions
    predictions = network.predict_batch(X_test[:16].tolist())
    y_pred = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test_images[i], cmap='gray')
        ax.set_title(f'True: {y_test[i]}, Pred: {y_pred[i]}\nConf: {confidences[i]:.3f}')
        ax.axis('off')
    
    plt.suptitle('MNIST Sample Predictions')
    plt.tight_layout()
    plt.savefig('mnist_sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()