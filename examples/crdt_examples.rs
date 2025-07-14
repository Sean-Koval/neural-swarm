//! CRDT examples for neuroplex distributed memory system

use neuroplex::{GCounter, PNCounter, ORSet, LWWRegister, init_tracing};
use std::error::Error;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize tracing
    init_tracing();
    
    println!("=== Neuroplex CRDT Examples ===");
    
    // G-Counter example
    println!("\n--- G-Counter (Grow-Only Counter) ---");
    let mut counter1 = GCounter::new();
    let mut counter2 = GCounter::new();
    
    let node1 = Uuid::new_v4();
    let node2 = Uuid::new_v4();
    
    // Increment counters on different nodes
    counter1.increment(node1, 5)?;
    counter1.increment(node1, 3)?;
    println!("Counter1 (node1): {}", counter1.count());
    
    counter2.increment(node2, 7)?;
    counter2.increment(node2, 2)?;
    println!("Counter2 (node2): {}", counter2.count());
    
    // Merge counters
    counter1.merge(&counter2)?;
    println!("Merged counter: {}", counter1.count());
    
    // PN-Counter example
    println!("\n--- PN-Counter (Increment/Decrement Counter) ---");
    let mut pn_counter1 = PNCounter::new();
    let mut pn_counter2 = PNCounter::new();
    
    // Operations on different nodes
    pn_counter1.increment(node1, 10)?;
    pn_counter1.decrement(node1, 3)?;
    println!("PN-Counter1 (node1): {}", pn_counter1.count());
    
    pn_counter2.increment(node2, 5)?;
    pn_counter2.decrement(node2, 2)?;
    println!("PN-Counter2 (node2): {}", pn_counter2.count());
    
    // Merge PN-counters
    pn_counter1.merge(&pn_counter2)?;
    println!("Merged PN-Counter: {}", pn_counter1.count());
    
    // OR-Set example
    println!("\n--- OR-Set (Observed-Remove Set) ---");
    let mut set1 = ORSet::new();
    let mut set2 = ORSet::new();
    
    // Add elements on different nodes
    set1.add("apple".to_string(), node1)?;
    set1.add("banana".to_string(), node1)?;
    println!("Set1 elements: {:?}", set1.elements());
    
    set2.add("banana".to_string(), node2)?;
    set2.add("cherry".to_string(), node2)?;
    println!("Set2 elements: {:?}", set2.elements());
    
    // Merge sets
    set1.merge(&set2)?;
    println!("Merged set: {:?}", set1.elements());
    
    // LWW-Register example
    println!("\n--- LWW-Register (Last-Writer-Wins Register) ---");
    let mut register1 = LWWRegister::new();
    let mut register2 = LWWRegister::new();
    
    // Set values on different nodes
    register1.set("initial_value".to_string(), node1)?;
    println!("Register1 value: {:?}", register1.get());
    
    // Simulate concurrent updates
    std::thread::sleep(std::time::Duration::from_millis(1));
    register2.set("updated_value".to_string(), node2)?;
    println!("Register2 value: {:?}", register2.get());
    
    // Merge registers (should resolve to the later value)
    register1.merge(&register2)?;
    println!("Merged register: {:?}", register1.get());
    
    // Demonstrate conflict resolution
    println!("\n--- Conflict Resolution Demo ---");
    let mut reg_a = LWWRegister::new();
    let mut reg_b = LWWRegister::new();
    
    // Simultaneous updates (same timestamp)
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    // These would normally use the internal timestamp mechanism
    reg_a.set("value_from_a".to_string(), node1)?;
    reg_b.set("value_from_b".to_string(), node2)?;
    
    println!("Before merge - A: {:?}, B: {:?}", reg_a.get(), reg_b.get());
    
    // Merge - should resolve deterministically
    reg_a.merge(&reg_b)?;
    println!("After merge - A: {:?}", reg_a.get());
    
    // Demonstrate version vectors
    println!("\n--- Version Vector Demo ---");
    let mut counter_demo = GCounter::new();
    
    // Increment from different nodes
    counter_demo.increment(node1, 1)?;
    counter_demo.increment(node2, 2)?;
    counter_demo.increment(node1, 3)?;
    
    let version_vector = counter_demo.version_vector();
    println!("Version vector: {:?}", version_vector);
    println!("Final count: {}", counter_demo.count());
    
    // Delta synchronization example
    println!("\n--- Delta Synchronization Demo ---");
    let mut base_counter = GCounter::new();
    let mut delta_counter = GCounter::new();
    
    // Build base state
    base_counter.increment(node1, 5)?;
    base_counter.increment(node2, 3)?;
    
    // Create delta state
    delta_counter.increment(node1, 2)?;
    delta_counter.increment(node2, 1)?;
    
    println!("Base counter: {}", base_counter.count());
    println!("Delta counter: {}", delta_counter.count());
    
    // Apply delta
    base_counter.merge(&delta_counter)?;
    println!("After delta sync: {}", base_counter.count());
    
    println!("\n=== CRDT Examples Complete ===");
    
    Ok(())
}