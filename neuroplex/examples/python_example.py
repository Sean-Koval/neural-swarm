#!/usr/bin/env python3
"""
Python example for neuroplex distributed memory system.

This example demonstrates how to use the neuroplex Python bindings
for distributed operations with CRDTs and consensus.
"""

import asyncio
import json
import time
from uuid import uuid4
from typing import Dict, Any, Optional

# Import neuroplex (this would be available after building the Python package)
try:
    import neuroplex
except ImportError:
    print("❌ neuroplex Python package not found. Build with: maturin develop")
    exit(1)


class NeuroPlexDemo:
    """Demo class for neuroplex operations."""
    
    def __init__(self, node_id: str, cluster_nodes: list):
        """Initialize the demo with node configuration."""
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.system = None
        
    async def start(self):
        """Start the neuroplex system."""
        print(f"🚀 Starting neuroplex system for node {self.node_id}")
        
        # Create configuration
        config = neuroplex.NeuroPlexConfig(
            node_id=self.node_id,
            cluster_nodes=self.cluster_nodes
        )
        
        # Create and start system
        self.system = neuroplex.NeuroPlexSystem(config)
        await self.system.start()
        
        print("✅ System started successfully")
        
    async def stop(self):
        """Stop the neuroplex system."""
        if self.system:
            await self.system.stop()
            print("✅ System stopped gracefully")
    
    async def demonstrate_crdts(self):
        """Demonstrate CRDT operations."""
        print("\n📊 Demonstrating CRDT operations:")
        
        store = self.system.distributed_store()
        
        # G-Counter example
        print("  🔢 G-Counter operations:")
        g_counter = neuroplex.CrdtFactory.create("GCounter", self.node_id, None)
        await store.set("g_counter", g_counter)
        
        # Simulate increments
        for i in range(1, 6):
            counter = await store.get("g_counter")
            # In a real implementation, we'd apply operations directly
            print(f"    Increment {i}: Counter value = {counter.value()}")
            await asyncio.sleep(0.1)
        
        # PN-Counter example
        print("  ➕➖ PN-Counter operations:")
        pn_counter = neuroplex.CrdtFactory.create("PNCounter", self.node_id, None)
        await store.set("pn_counter", pn_counter)
        
        # Simulate increments and decrements
        operations = [("increment", 10), ("decrement", 3), ("increment", 5)]
        for op, amount in operations:
            counter = await store.get("pn_counter")
            print(f"    {op.capitalize()} {amount}: Counter value = {counter.value()}")
            await asyncio.sleep(0.1)
        
        # OR-Set example
        print("  🗂️ OR-Set operations:")
        or_set = neuroplex.CrdtFactory.create("ORSet", self.node_id, None)
        await store.set("or_set", or_set)
        
        # Simulate adding items
        items = ["apple", "banana", "orange", "grape"]
        for item in items:
            current_set = await store.get("or_set")
            print(f"    Added '{item}': Set = {current_set.value()}")
            await asyncio.sleep(0.1)
        
        # LWW-Register example
        print("  📝 LWW-Register operations:")
        register = neuroplex.CrdtFactory.create("LWWRegister", self.node_id, "Initial value")
        await store.set("register", register)
        
        # Simulate value updates
        values = ["Updated value 1", "Updated value 2", "Final value"]
        for value in values:
            current_register = await store.get("register")
            print(f"    Set '{value}': Register = {current_register.value()}")
            await asyncio.sleep(0.1)
    
    async def demonstrate_memory_management(self):
        """Demonstrate memory management operations."""
        print("\n💾 Demonstrating memory management:")
        
        memory_manager = self.system.memory_manager()
        
        # Allocate memory
        test_data = b"This is test data for memory management demo"
        await memory_manager.allocate("test_block", test_data)
        print(f"  ✅ Allocated {len(test_data)} bytes")
        
        # Read data back
        read_data = await memory_manager.read("test_block")
        print(f"  📖 Read data: {read_data.decode()}")
        
        # Update data
        new_data = b"Updated test data with more content"
        await memory_manager.update("test_block", new_data)
        print(f"  🔄 Updated with {len(new_data)} bytes")
        
        # Get usage statistics
        stats = await memory_manager.usage_stats()
        print(f"  📊 Memory usage: {stats.total_used} bytes, {stats.block_count} blocks")
        
        # Force garbage collection
        await memory_manager.force_gc()
        print("  🗑️ Garbage collection completed")
    
    async def demonstrate_consensus(self):
        """Demonstrate consensus operations."""
        print("\n🗳️ Demonstrating consensus:")
        
        consensus = self.system.consensus_engine()
        
        # Get current state
        state = await consensus.state()
        print(f"  📊 Current state: {state}")
        
        # Get current term
        term = await consensus.term()
        print(f"  📅 Current term: {term}")
        
        # Get leader
        leader = await consensus.leader()
        print(f"  👑 Current leader: {leader}")
        
        # Propose a log entry
        try:
            log_index = await consensus.propose(b"Test log entry")
            print(f"  📝 Proposed log entry at index: {log_index}")
        except Exception as e:
            print(f"  ❌ Failed to propose (expected if not leader): {e}")
        
        # Get cluster status
        cluster_status = await consensus.cluster_status()
        print("  🏥 Cluster status:")
        for key, value in cluster_status.items():
            print(f"    {key}: {value}")
    
    async def demonstrate_synchronization(self):
        """Demonstrate synchronization operations."""
        print("\n🔄 Demonstrating synchronization:")
        
        sync_coordinator = self.system.sync_coordinator()
        
        # Sync with peers
        try:
            await sync_coordinator.sync_with_peers()
            print("  ✅ Synced with peers")
        except Exception as e:
            print(f"  ⚠️ Sync with peers (expected in single node): {e}")
        
        # Get sync status
        status = await sync_coordinator.status()
        print("  📊 Sync status:")
        for key, value in status.items():
            print(f"    {key}: {value}")
    
    async def monitor_events(self, duration: float = 5.0):
        """Monitor system events for a specified duration."""
        print(f"\n📡 Monitoring events for {duration} seconds:")
        
        store = self.system.distributed_store()
        
        # Subscribe to store events
        store_events = store.subscribe()
        
        # Monitor events
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                try:
                    # This is a simplified event monitoring
                    # In actual implementation, you'd receive actual events
                    await asyncio.sleep(0.5)
                    print("  🔍 Monitoring for store events...")
                except Exception as e:
                    print(f"  ❌ Event monitoring error: {e}")
                    break
        except KeyboardInterrupt:
            print("  ⏸️ Event monitoring interrupted")
        
        print("  ✅ Event monitoring completed")
    
    async def get_system_health(self):
        """Get and display system health information."""
        print("\n🏥 System health:")
        
        health = await self.system.health_status()
        for key, value in health.items():
            print(f"  {key}: {value}")
        
        return health


async def run_single_node_demo():
    """Run a single node demonstration."""
    print("🚀 Starting single node neuroplex demo")
    
    # Create demo instance
    demo = NeuroPlexDemo(
        node_id=str(uuid4()),
        cluster_nodes=["127.0.0.1:8000"]
    )
    
    try:
        # Start system
        await demo.start()
        
        # Run demonstrations
        await demo.demonstrate_crdts()
        await demo.demonstrate_memory_management()
        await demo.demonstrate_consensus()
        await demo.demonstrate_synchronization()
        
        # Monitor events
        await demo.monitor_events(3.0)
        
        # Get final health status
        health = await demo.get_system_health()
        
        print("\n📊 Demo Summary:")
        print(f"  Node ID: {demo.node_id}")
        print(f"  Cluster nodes: {demo.cluster_nodes}")
        print(f"  System health: {health.get('consensus_state', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop system
        await demo.stop()


async def run_multi_node_demo():
    """Run a multi-node demonstration."""
    print("🚀 Starting multi-node neuroplex demo")
    
    # Create multiple demo instances
    cluster_nodes = ["127.0.0.1:8000", "127.0.0.1:8001", "127.0.0.1:8002"]
    
    demos = [
        NeuroPlexDemo(str(uuid4()), cluster_nodes),
        NeuroPlexDemo(str(uuid4()), cluster_nodes),
        NeuroPlexDemo(str(uuid4()), cluster_nodes),
    ]
    
    try:
        # Start all nodes
        print("🔧 Starting all nodes...")
        for i, demo in enumerate(demos):
            await demo.start()
            print(f"  Node {i+1} started")
        
        # Wait for cluster formation
        print("⏳ Waiting for cluster formation...")
        await asyncio.sleep(2)
        
        # Run operations on all nodes
        print("🔄 Running operations on all nodes...")
        tasks = []
        for i, demo in enumerate(demos):
            task = asyncio.create_task(demo.demonstrate_crdts())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Check health of all nodes
        print("🏥 Checking health of all nodes:")
        for i, demo in enumerate(demos):
            print(f"  Node {i+1}:")
            health = await demo.get_system_health()
            print(f"    Status: {health.get('consensus_state', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Multi-node demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop all nodes
        print("🛑 Stopping all nodes...")
        for demo in demos:
            await demo.stop()


def main():
    """Main entry point."""
    print("🎯 Neuroplex Python Demo")
    print("=" * 50)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        asyncio.run(run_multi_node_demo())
    else:
        asyncio.run(run_single_node_demo())
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()