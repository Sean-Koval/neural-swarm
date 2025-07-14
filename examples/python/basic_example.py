#!/usr/bin/env python3
"""
Basic FANN-Rust-Core Python Example

This example demonstrates the fundamental usage of the Python bindings
for FANN-Rust-Core, including network creation, training, and evaluation.
"""

import fann_rust_core as fann
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

def main():
    print("FANN-Rust-Core Python: Basic Example")
    print("====================================")
    
    # Generate sample classification data
    print("📊 Generating sample classification dataset...")
    X, y = generate_sample_data()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Dataset created:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Create neural network
    print("\n🏗️  Creating neural network...")
    network = create_network(input_size=X_train.shape[1])
    
    # Train the network
    print("\n🚀 Training the network...")
    training_history = train_network(network, X_train, y_train)
    
    # Evaluate the network
    print("\n📊 Evaluating the network...")
    accuracy = evaluate_network(network, X_test, y_test)
    
    # Demonstrate advanced features
    print("\n⚡ Demonstrating advanced features...")
    demonstrate_advanced_features(network, X_test[:10])
    
    # Save and load model
    print("\n💾 Saving and loading model...")
    demonstrate_save_load(network, X_test[0])
    
    # Plot training history
    plot_training_history(training_history)
    
    print(f"\n✅ Example completed! Final accuracy: {accuracy:.4f}")

def generate_sample_data():
    """Generate a sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X.astype(np.float32), y

def create_network(input_size):
    """Create and configure a neural network."""
    network = fann.NeuralNetwork(
        layers=[input_size, 64, 32, 3],  # 3 classes
        activation='relu',
        output_activation='softmax',
        learning_rate=0.001,
        optimizer='adam',
        use_simd=True
    )
    
    print(f"✓ Created network: [{input_size}, 64, 32, 3]")
    print("   Activation: ReLU -> Softmax")
    print("   Optimizer: Adam (lr=0.001)")
    print("   SIMD optimization: Enabled")
    
    return network

def train_network(network, X_train, y_train):
    """Train the neural network and return training history."""
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y_train))
    y_train_onehot = np.eye(num_classes)[y_train].astype(np.float32)
    
    # Training callback to track progress
    training_history = {'epoch': [], 'loss': [], 'accuracy': []}
    
    def training_callback(epoch, loss, accuracy):
        training_history['epoch'].append(epoch)
        training_history['loss'].append(loss)
        training_history['accuracy'].append(accuracy)
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    start_time = time.time()
    
    # Train the network
    final_error = network.train(
        inputs=X_train.tolist(),
        targets=y_train_onehot.tolist(),
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        early_stopping=True,
        patience=15,
        callback=training_callback
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ Training completed!")
    print(f"   Final error: {final_error:.6f}")
    print(f"   Training time: {training_time:.2f}s")
    print(f"   Epochs trained: {len(training_history['epoch'])}")
    
    return training_history

def evaluate_network(network, X_test, y_test):
    """Evaluate the trained network."""
    # Make predictions
    predictions = network.predict_batch(X_test.tolist())
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Test accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("📊 Confusion Matrix:")
    print(cm)
    
    return accuracy

def demonstrate_advanced_features(network, sample_data):
    """Demonstrate advanced features of the library."""
    print("🔧 Performance monitoring...")
    
    # Benchmark inference speed
    start_time = time.time()
    for _ in range(1000):
        _ = network.predict(sample_data[0].tolist())
    single_inference_time = (time.time() - start_time) / 1000
    
    # Benchmark batch inference
    start_time = time.time()
    _ = network.predict_batch([row.tolist() for row in sample_data])
    batch_inference_time = time.time() - start_time
    
    print(f"   Single inference: {single_inference_time*1000:.2f}ms")
    print(f"   Batch inference (10 samples): {batch_inference_time*1000:.2f}ms")
    print(f"   Speedup from batching: {(single_inference_time*10/batch_inference_time):.1f}x")
    
    # Memory usage estimation
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   Current memory usage: {memory_usage:.1f}MB")
    except ImportError:
        print("   Memory monitoring requires psutil package")

def demonstrate_save_load(network, test_sample):
    """Demonstrate saving and loading models."""
    # Save the model
    network.save('example_model.fann')
    print("✓ Model saved to 'example_model.fann'")
    
    # Load the model
    loaded_network = fann.NeuralNetwork.load('example_model.fann')
    print("✓ Model loaded from file")
    
    # Verify consistency
    original_output = network.predict(test_sample.tolist())
    loaded_output = loaded_network.predict(test_sample.tolist())
    
    difference = np.abs(np.array(original_output) - np.array(loaded_output)).max()
    print(f"✓ Output consistency check: max difference = {difference:.8f}")
    
    if difference < 1e-6:
        print("   ✅ Loaded model produces identical outputs")
    else:
        print("   ⚠️  Small differences detected (normal for floating-point)")

def plot_training_history(history):
    """Plot training history."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history['epoch'], history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['epoch'], history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✓ Training history plotted and saved as 'training_history.png'")
        
    except ImportError:
        print("📊 Matplotlib not available - skipping plot generation")

def advanced_example():
    """Demonstrate more advanced features."""
    print("\n🔬 Advanced Features Example")
    print("============================")
    
    # Create a more complex network with custom configuration
    from fann_rust_core import NetworkBuilder, LayerConfig
    
    builder = NetworkBuilder()
    
    # Add layers with detailed configuration
    builder.add_layer(LayerConfig(
        size=20,
        activation='relu',
        dropout=0.0,
        batch_normalization=False
    ))
    
    builder.add_layer(LayerConfig(
        size=64,
        activation='relu',
        dropout=0.3,
        batch_normalization=True
    ))
    
    builder.add_layer(LayerConfig(
        size=32,
        activation='relu',
        dropout=0.5,
        batch_normalization=True
    ))
    
    builder.add_layer(LayerConfig(
        size=3,
        activation='softmax',
        dropout=0.0,
        batch_normalization=False
    ))
    
    # Configure optimizer and training
    builder.set_optimizer('adam', learning_rate=0.001, beta1=0.9, beta2=0.999)
    builder.set_loss_function('cross_entropy')
    builder.set_regularization(l1=0.0001, l2=0.001)
    
    # Build the network
    advanced_network = builder.build()
    print("✓ Created advanced network with custom configuration")
    
    return advanced_network

if __name__ == "__main__":
    main()
    
    # Run advanced example if time permits
    try:
        advanced_example()
    except Exception as e:
        print(f"Advanced example failed: {e}")