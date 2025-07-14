use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use fann_rust_core::{
    memory::{MemoryPool, AlignedVec, MemoryAnalyzer, AllocationPattern},
    profiling::{MemoryProfiler, AllocationTracker},
};
use std::time::Duration;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global memory usage tracker for profiling
struct TrackingAllocator {
    allocated: AtomicUsize,
    deallocated: AtomicUsize,
    peak_usage: AtomicUsize,
}

impl TrackingAllocator {
    const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            deallocated: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
        }
    }
    
    fn current_usage(&self) -> usize {
        self.allocated.load(Ordering::Relaxed) - self.deallocated.load(Ordering::Relaxed)
    }
    
    fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::Relaxed)
    }
    
    fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.deallocated.store(0, Ordering::Relaxed);
        self.peak_usage.store(0, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let allocated = self.allocated.fetch_add(size, Ordering::Relaxed) + size;
            let current = allocated - self.deallocated.load(Ordering::Relaxed);
            
            // Update peak usage
            let mut peak = self.peak_usage.load(Ordering::Relaxed);
            while current > peak {
                match self.peak_usage.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(new_peak) => peak = new_peak,
                }
            }
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.deallocated.fetch_add(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static TRACKING_ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

/// Benchmark memory pool allocation patterns
fn benchmark_memory_pools(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pools");
    
    let pool_sizes = vec![64, 256, 1024, 4096, 16384];
    let allocation_counts = vec![100, 1000, 10000];
    
    for &pool_size in &pool_sizes {
        for &alloc_count in &allocation_counts {
            group.throughput(Throughput::Elements(alloc_count as u64));
            
            // Benchmark standard allocation
            group.bench_with_input(
                BenchmarkId::new("std_alloc", format!("{}_{}", pool_size, alloc_count)),
                &(pool_size, alloc_count),
                |bench, &(size, count)| {
                    bench.iter_batched(
                        || {
                            TRACKING_ALLOCATOR.reset();
                            Vec::new()
                        },
                        |mut allocations: Vec<Vec<f32>>| {
                            for _ in 0..count {
                                let allocation = black_box(vec![0.0f32; size / 4]);
                                allocations.push(allocation);
                            }
                            let peak = TRACKING_ALLOCATOR.peak_usage();
                            black_box((allocations, peak));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Benchmark memory pool allocation
            group.bench_with_input(
                BenchmarkId::new("pool_alloc", format!("{}_{}", pool_size, alloc_count)),
                &(pool_size, alloc_count),
                |bench, &(size, count)| {
                    bench.iter_batched(
                        || {
                            TRACKING_ALLOCATOR.reset();
                            (MemoryPool::new(size, 32), Vec::new())
                        },
                        |(mut pool, mut allocations): (MemoryPool, Vec<*mut u8>)| {
                            for _ in 0..count {
                                if let Some(ptr) = pool.allocate() {
                                    allocations.push(black_box(ptr));
                                }
                            }
                            
                            // Cleanup
                            for ptr in allocations {
                                pool.deallocate(ptr);
                            }
                            
                            let peak = TRACKING_ALLOCATOR.peak_usage();
                            black_box(peak);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            
            // Benchmark aligned allocation
            group.bench_with_input(
                BenchmarkId::new("aligned_alloc", format!("{}_{}", pool_size, alloc_count)),
                &(pool_size, alloc_count),
                |bench, &(size, count)| {
                    bench.iter_batched(
                        || {
                            TRACKING_ALLOCATOR.reset();
                            Vec::new()
                        },
                        |mut allocations: Vec<AlignedVec<f32>>| {
                            for _ in 0..count {
                                let allocation = black_box(AlignedVec::<f32>::new_zeroed(size / 4, 32));
                                allocations.push(allocation);
                            }
                            let peak = TRACKING_ALLOCATOR.peak_usage();
                            black_box((allocations, peak));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark memory fragmentation patterns
fn benchmark_fragmentation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmentation_patterns");
    
    let patterns = vec![
        AllocationPattern::Sequential,
        AllocationPattern::Random,
        AllocationPattern::Batch,
        AllocationPattern::Fragmented,
    ];
    
    for pattern in &patterns {
        let pattern_name = format!("{:?}", pattern);
        
        group.bench_with_input(
            BenchmarkId::new("fragmentation_test", &pattern_name),
            pattern,
            |bench, pattern| {
                bench.iter_batched(
                    || {
                        TRACKING_ALLOCATOR.reset();
                        MemoryPool::new(1024, 32)
                    },
                    |mut pool| {
                        let analyzer = MemoryAnalyzer::new();
                        let fragmentation = analyzer.simulate_allocation_pattern(&mut pool, pattern, 1000);
                        black_box(fragmentation);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache-aware data structures
fn benchmark_cache_awareness(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_awareness");
    
    let sizes = vec![1024, 4096, 16384, 65536];
    
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Array of Structures (AoS) - cache unfriendly
        group.bench_with_input(
            BenchmarkId::new("aos_access", size),
            &size,
            |bench, &size| {
                #[derive(Clone)]
                struct Neuron {
                    weight: f32,
                    bias: f32,
                    activation: f32,
                    error: f32,
                }
                
                let neurons: Vec<Neuron> = (0..size).map(|i| Neuron {
                    weight: i as f32,
                    bias: (i as f32) * 0.1,
                    activation: 0.0,
                    error: 0.0,
                }).collect();
                
                bench.iter(|| {
                    let mut sum = 0.0f32;
                    for neuron in black_box(&neurons) {
                        sum += neuron.weight + neuron.bias;
                    }
                    black_box(sum);
                });
            },
        );
        
        // Structure of Arrays (SoA) - cache friendly
        group.bench_with_input(
            BenchmarkId::new("soa_access", size),
            &size,
            |bench, &size| {
                struct NeuronLayer {
                    weights: AlignedVec<f32>,
                    biases: AlignedVec<f32>,
                    activations: AlignedVec<f32>,
                    errors: AlignedVec<f32>,
                }
                
                let layer = NeuronLayer {
                    weights: AlignedVec::from_iter((0..size).map(|i| i as f32), 32),
                    biases: AlignedVec::from_iter((0..size).map(|i| (i as f32) * 0.1), 32),
                    activations: AlignedVec::new_zeroed(size, 32),
                    errors: AlignedVec::new_zeroed(size, 32),
                };
                
                bench.iter(|| {
                    let mut sum = 0.0f32;
                    for i in 0..black_box(&layer.weights).len() {
                        sum += layer.weights[i] + layer.biases[i];
                    }
                    black_box(sum);
                });
            },
        );
        
        // Packed structure access
        group.bench_with_input(
            BenchmarkId::new("packed_access", size),
            &size,
            |bench, &size| {
                #[repr(packed)]
                struct PackedNeuron {
                    weight: f32,
                    bias: f32,
                }
                
                let neurons: Vec<PackedNeuron> = (0..size).map(|i| PackedNeuron {
                    weight: i as f32,
                    bias: (i as f32) * 0.1,
                }).collect();
                
                bench.iter(|| {
                    let mut sum = 0.0f32;
                    for neuron in black_box(&neurons) {
                        sum += neuron.weight + neuron.bias;
                    }
                    black_box(sum);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory access patterns for neural network workloads
fn benchmark_neural_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_memory_patterns");
    
    // Simulate different neural network memory access patterns
    let network_sizes = vec![
        (784, 128, 10),    // MNIST-like
        (1000, 500, 100), // Medium network
        (2048, 1024, 512), // Large network
    ];
    
    for &(input_size, hidden_size, output_size) in &network_sizes {
        let network_name = format!("{}x{}x{}", input_size, hidden_size, output_size);
        
        // Forward pass memory pattern
        group.bench_with_input(
            BenchmarkId::new("forward_pass_memory", &network_name),
            &(input_size, hidden_size, output_size),
            |bench, &(input_size, hidden_size, output_size)| {
                let weights1 = AlignedVec::<f32>::from_iter(
                    (0..input_size * hidden_size).map(|_| rand::random::<f32>()),
                    32
                );
                let weights2 = AlignedVec::<f32>::from_iter(
                    (0..hidden_size * output_size).map(|_| rand::random::<f32>()),
                    32
                );
                let input = vec![1.0f32; input_size];
                
                bench.iter_batched(
                    || {
                        TRACKING_ALLOCATOR.reset();
                        (
                            vec![0.0f32; hidden_size],
                            vec![0.0f32; output_size],
                        )
                    },
                    |(mut hidden, mut output)| {
                        // Layer 1 computation
                        for i in 0..hidden_size {
                            hidden[i] = 0.0;
                            for j in 0..input_size {
                                hidden[i] += weights1[i * input_size + j] * input[j];
                            }
                            hidden[i] = hidden[i].max(0.0); // ReLU
                        }
                        
                        // Layer 2 computation
                        for i in 0..output_size {
                            output[i] = 0.0;
                            for j in 0..hidden_size {
                                output[i] += weights2[i * hidden_size + j] * hidden[j];
                            }
                        }
                        
                        let peak_memory = TRACKING_ALLOCATOR.peak_usage();
                        black_box((hidden, output, peak_memory));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        
        // Training memory pattern (with gradients)
        group.bench_with_input(
            BenchmarkId::new("training_memory", &network_name),
            &(input_size, hidden_size, output_size),
            |bench, &(input_size, hidden_size, output_size)| {
                let weights1 = AlignedVec::<f32>::from_iter(
                    (0..input_size * hidden_size).map(|_| rand::random::<f32>()),
                    32
                );
                let weights2 = AlignedVec::<f32>::from_iter(
                    (0..hidden_size * output_size).map(|_| rand::random::<f32>()),
                    32
                );
                let input = vec![1.0f32; input_size];
                let target = vec![1.0f32; output_size];
                
                bench.iter_batched(
                    || {
                        TRACKING_ALLOCATOR.reset();
                        (
                            vec![0.0f32; hidden_size],
                            vec![0.0f32; output_size],
                            vec![0.0f32; input_size * hidden_size],
                            vec![0.0f32; hidden_size * output_size],
                            vec![0.0f32; output_size],
                            vec![0.0f32; hidden_size],
                        )
                    },
                    |(mut hidden, mut output, mut grad_w1, mut grad_w2, mut grad_output, mut grad_hidden)| {
                        // Forward pass
                        for i in 0..hidden_size {
                            hidden[i] = 0.0;
                            for j in 0..input_size {
                                hidden[i] += weights1[i * input_size + j] * input[j];
                            }
                            hidden[i] = hidden[i].max(0.0); // ReLU
                        }
                        
                        for i in 0..output_size {
                            output[i] = 0.0;
                            for j in 0..hidden_size {
                                output[i] += weights2[i * hidden_size + j] * hidden[j];
                            }
                        }
                        
                        // Backward pass
                        for i in 0..output_size {
                            grad_output[i] = 2.0 * (output[i] - target[i]);
                        }
                        
                        for i in 0..hidden_size {
                            grad_hidden[i] = 0.0;
                            for j in 0..output_size {
                                grad_hidden[i] += grad_output[j] * weights2[j * hidden_size + i];
                            }
                            if hidden[i] <= 0.0 {
                                grad_hidden[i] = 0.0; // ReLU derivative
                            }
                        }
                        
                        // Compute weight gradients
                        for i in 0..hidden_size * output_size {
                            grad_w2[i] = grad_output[i / hidden_size] * hidden[i % hidden_size];
                        }
                        
                        for i in 0..input_size * hidden_size {
                            grad_w1[i] = grad_hidden[i / input_size] * input[i % input_size];
                        }
                        
                        let peak_memory = TRACKING_ALLOCATOR.peak_usage();
                        black_box((hidden, output, grad_w1, grad_w2, peak_memory));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory profiling overhead
fn benchmark_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_overhead");
    
    let operations = 10000;
    group.throughput(Throughput::Elements(operations as u64));
    
    // Baseline - no profiling
    group.bench_function("no_profiling", |bench| {
        bench.iter(|| {
            let mut allocations = Vec::new();
            for i in 0..operations {
                let allocation = black_box(vec![0.0f32; 256]);
                allocations.push(allocation);
                if i % 100 == 0 {
                    allocations.clear();
                }
            }
            black_box(allocations);
        });
    });
    
    // With memory profiling
    group.bench_function("with_profiling", |bench| {
        bench.iter(|| {
            let mut profiler = MemoryProfiler::new();
            let mut allocations = Vec::new();
            
            for i in 0..operations {
                profiler.before_allocation(256 * 4);
                let allocation = black_box(vec![0.0f32; 256]);
                profiler.after_allocation(allocation.as_ptr() as usize, 256 * 4);
                allocations.push(allocation);
                
                if i % 100 == 0 {
                    for allocation in &allocations {
                        profiler.before_deallocation(allocation.as_ptr() as usize);
                    }
                    allocations.clear();
                }
            }
            
            let report = profiler.generate_report();
            black_box((allocations, report));
        });
    });
    
    // With allocation tracking
    group.bench_function("with_tracking", |bench| {
        bench.iter(|| {
            let mut tracker = AllocationTracker::new();
            let mut allocations = Vec::new();
            
            for i in 0..operations {
                tracker.track_allocation(256 * 4);
                let allocation = black_box(vec![0.0f32; 256]);
                allocations.push(allocation);
                
                if i % 100 == 0 {
                    for _ in &allocations {
                        tracker.track_deallocation(256 * 4);
                    }
                    allocations.clear();
                }
            }
            
            let stats = tracker.get_statistics();
            black_box((allocations, stats));
        });
    });
    
    group.finish();
}

criterion_group! {
    name = memory_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100)
        .with_plots();
    targets = 
        benchmark_memory_pools,
        benchmark_fragmentation_patterns,
        benchmark_cache_awareness,
        benchmark_neural_memory_patterns,
        benchmark_profiling_overhead
}

criterion_main!(memory_benches);