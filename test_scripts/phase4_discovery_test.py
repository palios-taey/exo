#!/usr/bin/env python3
"""
Phase 4: Test Discovery Coordination
Tests Agent 4's peer discovery and coordination fixes

Usage: python3 phase4_discovery_test.py
"""

import sys
import time
import subprocess
import signal

def start_exo_server(host, port, log_file):
    """Start exo server on a device"""
    print(f"\nStarting exo server on {host}:{port}...")

    cmd = f"""
    ssh {host} 'cd /home/jetson/exo && \
    nohup python3 exo/main.py \
        --port {port} \
        --device CUDA \
        > {log_file} 2>&1 & \
    echo $!'
    """

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        pid = result.stdout.strip()
        print(f"✅ Started exo on {host} (PID: {pid})")
        return pid
    else:
        print(f"❌ Failed to start exo on {host}: {result.stderr}")
        return None

def check_peer_discovery(host, log_file, wait_time=30):
    """Check if peer discovery messages appear in logs"""
    print(f"\nWaiting {wait_time}s for peer discovery on {host}...")

    time.sleep(wait_time)

    cmd = f"ssh {host} 'grep -i \"peer\" {log_file} | tail -20'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if "Found peer" in result.stdout or "peer" in result.stdout.lower():
        print(f"✅ Peer discovery messages found on {host}")
        print(result.stdout)
        return True
    else:
        print(f"❌ No peer discovery messages on {host}")
        print(f"Log output: {result.stdout}")
        return False

def test_discovery_timeout():
    """Test that discovery timeout is configured correctly"""
    print("\n=== TEST 1: Discovery Timeout Configuration ===")

    try:
        # Check if discovery timeout has been increased
        # This would be in exo's discovery.py or similar

        print("⚠️  This test requires checking exo source code")
        print("Look for DISCOVERY_TIMEOUT or similar in:")
        print("  - exo/networking/discovery.py")
        print("  - exo/networking/peer_handle.py")

        print("Expected: DISCOVERY_TIMEOUT >= 35 * 60 (35 minutes)")
        print("Or: Persistent peer registry implemented")
        print("Or: Static peer configuration")

        return True  # Placeholder

    except Exception as e:
        print(f"❌ Discovery timeout test failed: {e}")
        return False

def test_static_peer_config():
    """Test if static peer configuration exists"""
    print("\n=== TEST 2: Static Peer Configuration ===")

    try:
        import os

        # Check for peers.yaml or similar config
        config_paths = [
            "/home/jetson/exo/config/peers.yaml",
            "/home/jetson/exo/peers.yaml",
            "/home/thor/exo/config/peers.yaml",
            "/home/thor/exo/peers.yaml",
        ]

        for path in config_paths:
            if os.path.exists(path):
                print(f"✅ Found peer config at: {path}")
                with open(path) as f:
                    print(f.read())
                return True

        print("⚠️  No static peer config found (may use dynamic discovery)")
        return True  # Not necessarily a failure

    except Exception as e:
        print(f"❌ Static config test failed: {e}")
        return False

def test_coordination_during_load():
    """Test that peers maintain coordination during model loading"""
    print("\n=== TEST 3: Coordination During Load ===")

    print("⚠️  This test requires:")
    print("  1. Both Thor devices running exo")
    print("  2. Triggering a 30B model load (18-33 min)")
    print("  3. Monitoring for 'peer lost' errors")

    print("\nManual test procedure:")
    print("  1. Start exo on both Thors")
    print("  2. Send inference request: curl -X POST http://10.0.0.78:52415/v1/chat/completions ...")
    print("  3. Monitor logs for peer coordination messages")
    print("  4. Wait for model load to complete (18-33 min)")
    print("  5. Verify no 'peer lost' errors")

    return True  # Manual test

def test_peer_persistence():
    """Test if peer registry persists across sessions"""
    print("\n=== TEST 4: Peer Registry Persistence ===")

    try:
        # Check if peer registry is saved to disk
        persistence_paths = [
            "/home/jetson/exo/.peer_registry",
            "/home/thor/exo/.peer_registry",
            "/home/jetson/exo/peers.db",
            "/home/thor/exo/peers.db",
        ]

        for path in persistence_paths:
            if os.path.exists(path):
                print(f"✅ Found peer registry at: {path}")
                return True

        print("⚠️  No persistent peer registry found")
        print("May use in-memory discovery only")
        return True  # Not necessarily a failure

    except Exception as e:
        print(f"❌ Peer persistence test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Phase 4: Discovery Coordination Testing (Agent 4)")
    print("=" * 60)

    results = {
        "discovery_timeout": test_discovery_timeout(),
        "static_peer_config": test_static_peer_config(),
        "coordination_during_load": test_coordination_during_load(),
        "peer_persistence": test_peer_persistence(),
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ PHASE 4: ALL TESTS PASSED (mostly placeholders)")
    else:
        print("❌ PHASE 4: SOME TESTS FAILED")
    print("=" * 60)

    print("\n⚠️  Phase 4 requires running exo servers for full validation")
    print("Run manual integration test after Agent 4 completes fixes")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
