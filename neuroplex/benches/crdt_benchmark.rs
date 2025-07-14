//! CRDT benchmarks for neuroplex
//!
//! This benchmark suite measures the performance of different CRDT operations
//! to ensure optimal performance characteristics.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuroplex::crdt::{CrdtFactory, CrdtOperation, CrdtValue};
use uuid::Uuid;

fn benchmark_g_counter(c: &mut Criterion) {
    let mut group = c.benchmark_group("g_counter");
    
    let node_id = Uuid::new_v4();
    
    // Benchmark counter creation
    group.bench_function("create", |b| {
        b.iter(|| {
            black_box(CrdtFactory::create("GCounter", node_id, None).unwrap())
        })
    });
    
    // Benchmark increment operations
    group.bench_function("increment", |b| {
        let mut counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
        let operation = CrdtOperation::GCounterIncrement { node_id, amount: 1 };
        
        b.iter(|| {
            black_box(counter.apply_operation(&operation).unwrap())
        })
    });
    
    // Benchmark merge operations
    group.bench_function("merge", |b| {
        let mut counter1 = CrdtFactory::create("GCounter", node_id, None).unwrap();
        let mut counter2 = CrdtFactory::create("GCounter", node_id, None).unwrap();
        
        // Apply some operations to create state
        counter1.apply_operation(&CrdtOperation::GCounterIncrement { node_id, amount: 100 }).unwrap();
        counter2.apply_operation(&CrdtOperation::GCounterIncrement { node_id, amount: 200 }).unwrap();
        
        b.iter(|| {
            black_box(counter1.merge(&counter2).unwrap())
        })
    });
    
    // Benchmark value retrieval
    group.bench_function("value", |b| {
        let mut counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
        counter.apply_operation(&CrdtOperation::GCounterIncrement { node_id, amount: 1000 }).unwrap();
        
        b.iter(|| {
            black_box(counter.value())
        })
    });
    
    group.finish();
}

fn benchmark_pn_counter(c: &mut Criterion) {
    let mut group = c.benchmark_group("pn_counter");
    
    let node_id = Uuid::new_v4();
    
    // Benchmark counter creation
    group.bench_function("create", |b| {
        b.iter(|| {
            black_box(CrdtFactory::create("PNCounter", node_id, None).unwrap())
        })
    });
    
    // Benchmark increment operations
    group.bench_function("increment", |b| {
        let mut counter = CrdtFactory::create("PNCounter", node_id, None).unwrap();
        let operation = CrdtOperation::PNCounterIncrement { node_id, amount: 1 };
        
        b.iter(|| {
            black_box(counter.apply_operation(&operation).unwrap())
        })
    });
    
    // Benchmark decrement operations
    group.bench_function("decrement", |b| {
        let mut counter = CrdtFactory::create("PNCounter", node_id, None).unwrap();
        let operation = CrdtOperation::PNCounterDecrement { node_id, amount: 1 };
        
        b.iter(|| {
            black_box(counter.apply_operation(&operation).unwrap())
        })
    });
    
    // Benchmark merge operations
    group.bench_function("merge", |b| {
        let mut counter1 = CrdtFactory::create("PNCounter", node_id, None).unwrap();
        let mut counter2 = CrdtFactory::create("PNCounter", node_id, None).unwrap();
        
        // Apply some operations to create state
        counter1.apply_operation(&CrdtOperation::PNCounterIncrement { node_id, amount: 100 }).unwrap();
        counter2.apply_operation(&CrdtOperation::PNCounterDecrement { node_id, amount: 50 }).unwrap();
        
        b.iter(|| {
            black_box(counter1.merge(&counter2).unwrap())
        })
    });
    
    group.finish();
}

fn benchmark_or_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("or_set");
    
    let node_id = Uuid::new_v4();
    
    // Benchmark set creation
    group.bench_function("create", |b| {
        b.iter(|| {
            black_box(CrdtFactory::create("ORSet", node_id, None).unwrap())
        })
    });
    
    // Benchmark add operations
    group.bench_function("add", |b| {
        let mut set = CrdtFactory::create("ORSet", node_id, None).unwrap();
        let mut counter = 0;
        
        b.iter(|| {
            let operation = CrdtOperation::ORSetAdd { 
                element: format!("element_{}", counter), 
                node_id, 
                timestamp: counter as u64 
            };
            counter += 1;
            black_box(set.apply_operation(&operation).unwrap())
        })
    });
    
    // Benchmark remove operations
    group.bench_function("remove", |b| {
        let mut set = CrdtFactory::create("ORSet", node_id, None).unwrap();
        
        // Pre-populate set
        for i in 0..1000 {
            let operation = CrdtOperation::ORSetAdd { 
                element: format!("element_{}", i), 
                node_id, 
                timestamp: i as u64 
            };
            set.apply_operation(&operation).unwrap();
        }
        
        let mut counter = 0;
        b.iter(|| {
            let operation = CrdtOperation::ORSetRemove { 
                element: format!("element_{}", counter % 1000), 
                node_id, 
                timestamp: (counter + 1000) as u64 
            };
            counter += 1;
            black_box(set.apply_operation(&operation).unwrap())
        })
    });
    
    // Benchmark merge operations
    group.bench_function("merge", |b| {
        let mut set1 = CrdtFactory::create("ORSet", node_id, None).unwrap();
        let mut set2 = CrdtFactory::create("ORSet", node_id, None).unwrap();
        
        // Pre-populate sets
        for i in 0..100 {
            let operation1 = CrdtOperation::ORSetAdd { 
                element: format!("element_{}", i), 
                node_id, 
                timestamp: i as u64 
            };
            let operation2 = CrdtOperation::ORSetAdd { 
                element: format!("element_{}", i + 100), 
                node_id, 
                timestamp: (i + 100) as u64 
            };
            
            set1.apply_operation(&operation1).unwrap();
            set2.apply_operation(&operation2).unwrap();
        }
        
        b.iter(|| {
            black_box(set1.merge(&set2).unwrap())
        })
    });
    
    group.finish();
}

fn benchmark_lww_register(c: &mut Criterion) {
    let mut group = c.benchmark_group("lww_register");
    
    let node_id = Uuid::new_v4();
    
    // Benchmark register creation
    group.bench_function("create", |b| {
        b.iter(|| {
            black_box(CrdtFactory::create("LWWRegister", node_id, 
                Some(serde_json::Value::String("test".to_string()))).unwrap())
        })
    });
    
    // Benchmark set operations
    group.bench_function("set", |b| {
        let mut register = CrdtFactory::create("LWWRegister", node_id, 
            Some(serde_json::Value::String("initial".to_string()))).unwrap();
        let mut counter = 0;
        
        b.iter(|| {
            let operation = CrdtOperation::LWWRegisterSet { 
                value: format!("value_{}", counter), 
                node_id, 
                timestamp: counter as u64 
            };
            counter += 1;
            black_box(register.apply_operation(&operation).unwrap())
        })
    });
    
    // Benchmark merge operations
    group.bench_function("merge", |b| {
        let register1 = CrdtFactory::create("LWWRegister", node_id, 
            Some(serde_json::Value::String("value1".to_string()))).unwrap();
        let register2 = CrdtFactory::create("LWWRegister", node_id, 
            Some(serde_json::Value::String("value2".to_string()))).unwrap();
        
        b.iter(|| {
            black_box(register1.merge(&register2).unwrap())
        })
    });
    
    group.finish();
}

fn benchmark_crdt_factory(c: &mut Criterion) {
    let mut group = c.benchmark_group("crdt_factory");
    
    let node_id = Uuid::new_v4();
    
    // Benchmark different CRDT type creation
    let crdt_types = vec!["GCounter", "PNCounter", "ORSet", "LWWRegister"];
    
    for crdt_type in crdt_types {
        group.bench_with_input(BenchmarkId::new("create", crdt_type), crdt_type, |b, &crdt_type| {
            b.iter(|| {
                let initial_value = if crdt_type == "LWWRegister" {
                    Some(serde_json::Value::String("test".to_string()))
                } else {
                    None
                };
                black_box(CrdtFactory::create(crdt_type, node_id, initial_value).unwrap())
            })
        });
    }
    
    group.finish();
}

fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    
    let node_id = Uuid::new_v4();
    
    // Create different CRDT types with some state
    let mut g_counter = CrdtFactory::create("GCounter", node_id, None).unwrap();
    g_counter.apply_operation(&CrdtOperation::GCounterIncrement { node_id, amount: 1000 }).unwrap();
    
    let mut pn_counter = CrdtFactory::create("PNCounter", node_id, None).unwrap();
    pn_counter.apply_operation(&CrdtOperation::PNCounterIncrement { node_id, amount: 1000 }).unwrap();
    pn_counter.apply_operation(&CrdtOperation::PNCounterDecrement { node_id, amount: 500 }).unwrap();
    
    let mut or_set = CrdtFactory::create("ORSet", node_id, None).unwrap();
    for i in 0..100 {
        or_set.apply_operation(&CrdtOperation::ORSetAdd { 
            element: format!("element_{}", i), 
            node_id, 
            timestamp: i as u64 
        }).unwrap();
    }
    
    let lww_register = CrdtFactory::create("LWWRegister", node_id, 
        Some(serde_json::Value::String("test_value".to_string()))).unwrap();
    
    // Benchmark serialization
    group.bench_function("g_counter_serialize", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(&g_counter).unwrap())
        })
    });
    
    group.bench_function("pn_counter_serialize", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(&pn_counter).unwrap())
        })
    });
    
    group.bench_function("or_set_serialize", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(&or_set).unwrap())
        })
    });
    
    group.bench_function("lww_register_serialize", |b| {
        b.iter(|| {
            black_box(serde_json::to_string(&lww_register).unwrap())
        })
    });
    
    // Benchmark deserialization
    let g_counter_json = serde_json::to_string(&g_counter).unwrap();
    let pn_counter_json = serde_json::to_string(&pn_counter).unwrap();
    let or_set_json = serde_json::to_string(&or_set).unwrap();
    let lww_register_json = serde_json::to_string(&lww_register).unwrap();
    
    group.bench_function("g_counter_deserialize", |b| {
        b.iter(|| {
            black_box(serde_json::from_str::<CrdtValue>(&g_counter_json).unwrap())
        })
    });
    
    group.bench_function("pn_counter_deserialize", |b| {
        b.iter(|| {
            black_box(serde_json::from_str::<CrdtValue>(&pn_counter_json).unwrap())
        })
    });
    
    group.bench_function("or_set_deserialize", |b| {
        b.iter(|| {
            black_box(serde_json::from_str::<CrdtValue>(&or_set_json).unwrap())
        })
    });
    
    group.bench_function("lww_register_deserialize", |b| {
        b.iter(|| {
            black_box(serde_json::from_str::<CrdtValue>(&lww_register_json).unwrap())
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_g_counter,
    benchmark_pn_counter,
    benchmark_or_set,
    benchmark_lww_register,
    benchmark_crdt_factory,
    benchmark_serialization
);

criterion_main!(benches);