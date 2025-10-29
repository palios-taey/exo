#!/usr/bin/env python3
"""
Test 3: Distributed Initialization
Tests that two exo nodes can discover each other and establish connection.
Does NOT attempt inference, only tests peer discovery and communication.
"""

import sys
import os
import time
import signal
import subprocess
import json
import requests
from datetime import datetime
from pathlib import Path

# Thor device configuration
THOR1_IP = "10.0.0.93"
THOR1_USER = "jetson"
THOR1_PASSWORD = "papaDons1001s$"
THOR1_PORT = 52416

THOR2_IP = "10.0.0.78"
THOR2_USER = "thor"
THOR2_PASSWORD = "papaDons1001s$"
THOR2_PORT = 52415

def print_header(message):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70)

def print_success(message):
    """Print success message"""
    print(f"✓ {message}")

def print_failure(message):
    """Print failure message"""
    print(f"✗ {message}")

def print_info(key, value):
    """Print info key-value pair"""
    print(f"  {key}: {value}")

def ssh_execute(host, user, password, command, timeout=30):
    """Execute command via SSH using sshpass"""
    try:
        full_command = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no {user}@{host} '{command}'"
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)

def check_thor_available(host, user, password):
    """Check if Thor device is reachable"""
    returncode, stdout, stderr = ssh_execute(host, user, password, "echo 'test'", timeout=10)
    return returncode == 0

def kill_exo_on_thor(host, user, password, device_name):
    """Kill all exo processes on Thor device"""
    print_info(f"{device_name}", "Stopping any running exo processes...")

    # Kill exo processes
    cmd = "pkill -9 -f 'exo/main.py' || true"
    returncode, stdout, stderr = ssh_execute(host, user, password, cmd)

    # Wait a moment
    time.sleep(2)

    # Verify no processes remain
    cmd = "pgrep -f 'exo/main.py' || echo 'none'"
    returncode, stdout, stderr = ssh_execute(host, user, password, cmd)

    if stdout.strip() == 'none':
        print_success(f"{device_name} clean")
    else:
        print_failure(f"{device_name} still has exo processes: {stdout.strip()}")

def start_exo_on_thor(host, user, password, port, device_name, exo_path="/home/thor/exo"):
    """Start exo server on Thor device"""
    print_header(f"Starting Exo on {device_name}")

    log_file = f"/tmp/exo_{device_name.lower().replace(' ', '_')}.log"

    # Build command
    cmd = f"""cd {exo_path} && \
DEVICE=CUDA DEBUG=2 nohup python3 exo/main.py \
--inference-engine tinygrad \
--chatgpt-api-port {port} \
--node-port {port} \
--disable-tui \
> {log_file} 2>&1 & \
echo $!"""

    print_info(f"{device_name} command", cmd.replace('\n', ' '))

    returncode, stdout, stderr = ssh_execute(host, user, password, cmd, timeout=10)

    if returncode == 0:
        pid = stdout.strip()
        print_success(f"{device_name} started with PID {pid}")
        print_info("Log file", f"{host}:{log_file}")
        return pid
    else:
        print_failure(f"{device_name} failed to start")
        print_info("Error", stderr)
        return None

def wait_for_server_ready(host, port, device_name, timeout=60):
    """Wait for server to respond to health checks"""
    print_info(f"{device_name}", f"Waiting for server on {host}:{port}...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print_success(f"{device_name} ready after {elapsed:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(3)

    print_failure(f"{device_name} not ready after {timeout}s")
    return False

def check_peer_discovery(host, user, password, device_name, log_file):
    """Check if peer discovery succeeded by examining logs"""
    print_info(f"{device_name}", "Checking for peer discovery...")

    # Get last 100 lines of log
    cmd = f"tail -100 {log_file}"
    returncode, stdout, stderr = ssh_execute(host, user, password, cmd)

    if returncode != 0:
        print_failure(f"Failed to read log from {device_name}")
        return False

    # Look for peer discovery messages
    log_lines = stdout.split('\n')

    # Search for peer discovery patterns
    found_peer = False
    peer_info = []

    for line in log_lines:
        if 'Found peer' in line or 'Discovered peer' in line or 'Connected to peer' in line:
            found_peer = True
            peer_info.append(line.strip())

    if found_peer:
        print_success(f"{device_name} discovered peers")
        for info in peer_info[-3:]:  # Show last 3 peer messages
            print_info("Peer discovery", info)
        return True
    else:
        print_failure(f"{device_name} no peer discovery found in logs")
        return False

def check_grpc_connections(host, user, password, device_name):
    """Check if gRPC connections are established"""
    print_info(f"{device_name}", "Checking gRPC connections...")

    # Look for established gRPC connections
    cmd = "netstat -tn | grep ESTABLISHED | grep -E ':(5678|52415|52416)' || echo 'none'"
    returncode, stdout, stderr = ssh_execute(host, user, password, cmd)

    connections = stdout.strip()

    if connections and connections != 'none':
        print_success(f"{device_name} has gRPC connections")
        for conn in connections.split('\n')[:5]:  # Show first 5
            print_info("Connection", conn)
        return True
    else:
        print_failure(f"{device_name} no gRPC connections found")
        return False

def verify_both_gpus_detected(thor1_host, thor1_user, thor1_pass, thor2_host, thor2_user, thor2_pass):
    """Verify both GPUs are detected and initialized"""
    print_header("Verifying GPU Detection")

    results = {}

    for device_name, host, user, password in [
        ("Thor #1", thor1_host, thor1_user, thor1_pass),
        ("Thor #2", thor2_host, thor2_user, thor2_pass)
    ]:
        cmd = "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
        returncode, stdout, stderr = ssh_execute(host, user, password, cmd)

        if returncode == 0:
            gpu_info = stdout.strip()
            print_info(f"{device_name} GPU", gpu_info)
            results[device_name] = True
        else:
            print_failure(f"{device_name} GPU check failed")
            results[device_name] = False

    return all(results.values())

def show_log_tail(host, user, password, device_name, log_file, lines=30):
    """Show last N lines of log"""
    print_header(f"{device_name} Log (last {lines} lines)")

    cmd = f"tail -{lines} {log_file}"
    returncode, stdout, stderr = ssh_execute(host, user, password, cmd)

    if returncode == 0:
        print(stdout)
    else:
        print_failure(f"Failed to read log: {stderr}")

def main():
    """Run distributed initialization test"""
    print("\n" + "=" * 70)
    print("  EXO DISTRIBUTED INFERENCE - TEST 3: DISTRIBUTED INIT")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Pre-flight checks
    print_header("Pre-Flight Checks")

    # Check Thor availability
    print_info("Status", "Checking Thor device availability...")
    thor1_available = check_thor_available(THOR1_IP, THOR1_USER, THOR1_PASSWORD)
    thor2_available = check_thor_available(THOR2_IP, THOR2_USER, THOR2_PASSWORD)

    if not thor1_available:
        print_failure(f"Thor #1 ({THOR1_IP}) not reachable")
        return 1

    if not thor2_available:
        print_failure(f"Thor #2 ({THOR2_IP}) not reachable")
        return 1

    print_success("Both Thor devices reachable")

    # Cleanup existing processes
    kill_exo_on_thor(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1")
    kill_exo_on_thor(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2")

    time.sleep(3)

    # Verify GPUs detected
    verify_both_gpus_detected(
        THOR1_IP, THOR1_USER, THOR1_PASSWORD,
        THOR2_IP, THOR2_USER, THOR2_PASSWORD
    )

    # Start servers
    pid1 = start_exo_on_thor(THOR1_IP, THOR1_USER, THOR1_PASSWORD, THOR1_PORT, "Thor #1", "/home/jetson/exo")
    if not pid1:
        return 1

    time.sleep(5)  # Wait before starting second server

    pid2 = start_exo_on_thor(THOR2_IP, THOR2_USER, THOR2_PASSWORD, THOR2_PORT, "Thor #2", "/home/thor/exo")
    if not pid2:
        # Cleanup first server
        ssh_execute(THOR1_IP, THOR1_USER, THOR1_PASSWORD, f"kill -9 {pid1}")
        return 1

    try:
        # Wait for servers to be ready
        print_header("Waiting for Servers")

        ready1 = wait_for_server_ready(THOR1_IP, THOR1_PORT, "Thor #1", timeout=60)
        ready2 = wait_for_server_ready(THOR2_IP, THOR2_PORT, "Thor #2", timeout=60)

        if not (ready1 and ready2):
            print_failure("One or both servers did not start")
            return 1

        # Give time for peer discovery (UDP broadcasts)
        print_info("Status", "Waiting 15s for peer discovery...")
        time.sleep(15)

        # Run tests
        print_header("Running Tests")

        results = {
            "Thor #1 Peer Discovery": check_peer_discovery(
                THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1",
                "/tmp/exo_thor_#1.log"
            ),
            "Thor #2 Peer Discovery": check_peer_discovery(
                THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2",
                "/tmp/exo_thor_#2.log"
            ),
            "Thor #1 gRPC Connections": check_grpc_connections(
                THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1"
            ),
            "Thor #2 gRPC Connections": check_grpc_connections(
                THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2"
            ),
        }

        # Show logs
        show_log_tail(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1", "/tmp/exo_thor_#1.log")
        show_log_tail(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2", "/tmp/exo_thor_#2.log")

        # Print summary
        print_header("Test Summary")
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        failed = total - passed

        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            symbol = "✓" if result else "✗"
            print(f"{symbol} {test_name}: {status}")

        print(f"\nTotal: {total} tests")
        print(f"Passed: {passed} tests")
        print(f"Failed: {failed} tests")

        if failed == 0:
            print("\n✓ DISTRIBUTED INITIALIZATION TEST PASSED")
            print("  Both nodes discovered each other and established connections")
            return 0
        else:
            print(f"\n✗ {failed} TEST(S) FAILED")
            print("  Check logs above for peer discovery issues")
            return 1

    finally:
        # Cleanup
        print_header("Cleanup")
        print_info("Status", "Stopping servers...")

        kill_exo_on_thor(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1")
        kill_exo_on_thor(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2")

        print_success("Cleanup complete")

if __name__ == "__main__":
    sys.exit(main())
