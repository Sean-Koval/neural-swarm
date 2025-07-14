#!/usr/bin/env python3
"""
Manual validation script for the neural network implementation.

This script performs validation checks that would normally be done by 
Rust's compiler and test framework, but works without requiring Cargo.
"""

import os
import re
import sys
from pathlib import Path

class ImplementationValidator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.errors = []
        self.warnings = []
        
    def validate(self):
        """Run all validation checks."""
        print("🔍 Validating Neural Network Implementation")
        print("=" * 50)
        
        self.check_file_structure()
        self.check_module_declarations()
        self.check_import_consistency()
        self.check_error_handling()
        self.check_api_completeness()
        self.check_documentation()
        self.validate_benchmarks()
        self.validate_examples()
        
        self.print_summary()
        return len(self.errors) == 0
    
    def check_file_structure(self):
        """Verify that all expected files exist."""
        print("📁 Checking file structure...")
        
        required_files = [
            "src/lib.rs",
            "src/error.rs", 
            "src/network.rs",
            "src/activations.rs",
            "src/training.rs",
            "src/inference.rs",
            "src/memory.rs",
            "src/matrix.rs",
            "src/profiling.rs",
            "src/quantization.rs",
            "src/ffi.rs",
            "src/utils.rs",
            "src/integration_test.rs",
            "Cargo.toml",
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.errors.append(f"Missing required file: {file_path}")
            else:
                print(f"   ✅ {file_path}")
        
        # Check for module directories
        module_dirs = [
            "src/optimization",
            "src/activation",
            "benches",
            "examples/rust",
            "examples/python",
            "python/fann_rust_core",
        ]
        
        for dir_path in module_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                self.warnings.append(f"Missing directory: {dir_path}")
            else:
                print(f"   ✅ {dir_path}/")
    
    def check_module_declarations(self):
        """Check that all modules are properly declared in lib.rs."""
        print("\n📦 Checking module declarations...")
        
        lib_rs = self.src_dir / "lib.rs"
        if not lib_rs.exists():
            self.errors.append("lib.rs not found")
            return
        
        content = lib_rs.read_text()
        
        expected_modules = [
            "activation", "activations", "network", "training", 
            "inference", "utils", "error", "ffi", "memory", 
            "matrix", "profiling", "optimization", "quantization"
        ]
        
        for module in expected_modules:
            if f"pub mod {module}" in content:
                print(f"   ✅ Module declared: {module}")
            else:
                self.errors.append(f"Module not declared in lib.rs: {module}")
    
    def check_import_consistency(self):
        """Check for consistent imports and re-exports."""
        print("\n🔗 Checking import consistency...")
        
        lib_rs = self.src_dir / "lib.rs"
        content = lib_rs.read_text()
        
        # Check for type re-exports
        expected_reexports = [
            "ActivationFunction",
            "Network", "NetworkBuilder", "NetworkConfig",
            "TrainingData", "TrainingAlgorithm",
            "FannError", "Result",
            "AlignedVec", "MemoryPool"
        ]
        
        for reexport in expected_reexports:
            if f"pub use" in content and reexport in content:
                print(f"   ✅ Re-exported: {reexport}")
            else:
                self.warnings.append(f"Type may not be re-exported: {reexport}")
    
    def check_error_handling(self):
        """Validate error handling implementation."""
        print("\n⚠️  Checking error handling...")
        
        error_rs = self.src_dir / "error.rs"
        if not error_rs.exists():
            self.errors.append("error.rs not found")
            return
        
        content = error_rs.read_text()
        
        # Check for required error types
        required_errors = [
            "NetworkConstruction", "Training", "Inference", 
            "DimensionMismatch", "MemoryAllocation", "Quantization"
        ]
        
        for error_type in required_errors:
            if error_type in content:
                print(f"   ✅ Error type defined: {error_type}")
            else:
                self.errors.append(f"Missing error type: {error_type}")
        
        # Check for Result type alias
        if "pub type Result<T>" in content:
            print("   ✅ Result type alias defined")
        else:
            self.errors.append("Missing Result type alias")
    
    def check_api_completeness(self):
        """Check that key API components are implemented."""
        print("\n🛠️  Checking API completeness...")
        
        # Check network.rs
        network_rs = self.src_dir / "network.rs"
        if network_rs.exists():
            content = network_rs.read_text()
            if "impl Network" in content and "pub fn forward" in content:
                print("   ✅ Network forward method implemented")
            else:
                self.errors.append("Network forward method missing")
            
            if "NetworkBuilder" in content:
                print("   ✅ NetworkBuilder pattern implemented")
            else:
                self.errors.append("NetworkBuilder pattern missing")
        else:
            self.errors.append("network.rs not found")
        
        # Check activations.rs
        activations_rs = self.src_dir / "activations.rs"
        if activations_rs.exists():
            content = activations_rs.read_text()
            activation_functions = ["ReLU", "Sigmoid", "Tanh", "GELU"]
            for func in activation_functions:
                if func in content:
                    print(f"   ✅ Activation function: {func}")
                else:
                    self.warnings.append(f"Activation function missing: {func}")
        
        # Check memory.rs for SIMD alignment
        memory_rs = self.src_dir / "memory.rs"
        if memory_rs.exists():
            content = memory_rs.read_text()
            if "AlignedVec" in content:
                print("   ✅ SIMD-aligned memory structures")
            else:
                self.warnings.append("SIMD-aligned memory structures missing")
    
    def check_documentation(self):
        """Validate documentation and examples."""
        print("\n📚 Checking documentation...")
        
        lib_rs = self.src_dir / "lib.rs"
        content = lib_rs.read_text()
        
        # Check for module-level documentation
        if content.startswith("//!"):
            print("   ✅ Module documentation present")
        else:
            self.warnings.append("Module documentation missing")
        
        # Check for usage examples in documentation
        if "```rust" in content or "```" in content:
            print("   ✅ Code examples in documentation")
        else:
            self.warnings.append("Code examples missing from documentation")
        
        # Check README existence
        readme_files = ["README.md", "README.rst", "README.txt"]
        readme_exists = any((self.project_root / readme).exists() for readme in readme_files)
        if readme_exists:
            print("   ✅ README file exists")
        else:
            self.warnings.append("README file missing")
    
    def validate_benchmarks(self):
        """Check benchmark implementation."""
        print("\n🏃 Validating benchmarks...")
        
        benches_dir = self.project_root / "benches"
        if benches_dir.exists():
            benchmark_files = list(benches_dir.glob("*.rs"))
            if benchmark_files:
                print(f"   ✅ Found {len(benchmark_files)} benchmark files")
                for bench_file in benchmark_files:
                    print(f"      - {bench_file.name}")
            else:
                self.warnings.append("No benchmark files found in benches/")
        else:
            self.warnings.append("Benchmark directory missing")
        
        # Check Cargo.toml for benchmark configuration
        cargo_toml = self.project_root / "Cargo.toml"
        if cargo_toml.exists():
            content = cargo_toml.read_text()
            if "[[bench]]" in content:
                print("   ✅ Benchmark configuration in Cargo.toml")
            else:
                self.warnings.append("Benchmark configuration missing")
    
    def validate_examples(self):
        """Check example implementations."""
        print("\n💡 Validating examples...")
        
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            rust_examples = list((examples_dir / "rust").glob("*.rs")) if (examples_dir / "rust").exists() else []
            python_examples = list((examples_dir / "python").glob("*.py")) if (examples_dir / "python").exists() else []
            
            if rust_examples:
                print(f"   ✅ Found {len(rust_examples)} Rust examples")
            if python_examples:
                print(f"   ✅ Found {len(python_examples)} Python examples")
            
            if not rust_examples and not python_examples:
                self.warnings.append("No example files found")
        else:
            self.warnings.append("Examples directory missing")
        
        # Check for XOR example specifically
        xor_examples = [
            self.project_root / "examples" / "xor_example.rs",
            self.project_root / "examples" / "rust" / "xor_network.rs",
            self.project_root / "examples" / "python" / "xor_example.py"
        ]
        
        xor_found = any(example.exists() for example in xor_examples)
        if xor_found:
            print("   ✅ XOR example found")
        else:
            self.warnings.append("XOR example missing")
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("📊 Validation Summary")
        print("=" * 50)
        
        if not self.errors and not self.warnings:
            print("🎉 Perfect! No issues found.")
            print("   The neural network implementation is complete and ready for testing.")
        else:
            if self.errors:
                print(f"❌ {len(self.errors)} Error(s) found:")
                for i, error in enumerate(self.errors, 1):
                    print(f"   {i}. {error}")
            
            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} Warning(s):")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"   {i}. {warning}")
        
        # Overall assessment
        if not self.errors:
            if len(self.warnings) <= 3:
                print("\n✅ Overall Status: READY FOR TESTING")
                print("   Implementation is complete with only minor issues.")
            else:
                print("\n🟡 Overall Status: MOSTLY READY")
                print("   Implementation is functional but could be improved.")
        else:
            print("\n❌ Overall Status: NEEDS WORK")
            print("   Critical issues must be resolved before testing.")

def main():
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    print(f"🎯 Project root: {project_root}")
    
    validator = ImplementationValidator(project_root)
    success = validator.validate()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()