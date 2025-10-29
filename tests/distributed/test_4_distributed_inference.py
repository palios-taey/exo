#!/usr/bin/env python3
"""
Test 4: Full Distributed Inference
Tests complete distributed inference across two Thor devices.
This is the end-to-end test that validates the entire system.
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

# Model to test
TEST_MODEL = "llama-3.2-1b"  # Smallest model for quick testing
TEST_PROMPT = "What is the capital of France?"
MAX_TOKENS = 30

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
    """Execute command via SSH"""
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

def kill_exo_on_thor(host, user, password, device_name):
    """Kill all exo processes on Thor device"""
    print_info(f"{device_name}", "Stopping any running exo processes...")
    cmd = "pkill -9 -f 'exo/main.py' || true"
    ssh_execute(host, user, password, cmd)
    time.sleep(2)

def start_exo_on_thor(host, user, password, port, device_name, exo_path, model_name):
    """Start exo server on Thor device"""
    print_header(f"Starting {device_name}")

    log_file = f"/tmp/exo_{device_name.lower().replace(' ', '_').replace('#', '')}.log"

    cmd = f"""cd {exo_path} && \
DEVICE=CUDA DEBUG=3 nohup python3 exo/main.py \
--inference-engine tinygrad \
--default-model {model_name} \
--chatgpt-api-port {port} \
--node-port {port} \
--disable-tui \
> {log_file} 2>&1 & \
echo $!"""

    print_info("Command", cmd.replace('\n', ' '))

    returncode, stdout, stderr = ssh_execute(host, user, password, cmd, timeout=10)

    if returncode == 0:
        pid = stdout.strip()
        print_success(f"Started with PID {pid}")
        print_info("Log file", f"{host}:{log_file}")
        return pid, log_file
    else:
        print_failure(f"Failed to start: {stderr}")
        return None, None

def wait_for_server_ready(host, port, device_name, timeout=90):
    """Wait for server to be ready"""
    print_info(f"{device_name}", f"Waiting for http://{host}:{port}/health...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print_success(f"Ready after {elapsed:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(3)

    print_failure(f"Not ready after {timeout}s")
    return False

def monitor_gpu_usage(host, user, password, device_name):
    """Monitor GPU usage on device"""
    cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
    returncode, stdout, stderr = ssh_execute(host, user, password, cmd, timeout=5)

    if returncode == 0:
        util, mem = stdout.strip().split(', ')
        print_info(f"{device_name} GPU", f"Util: {util}%, Memory: {mem} MiB")
        return int(util), int(mem)
    else:
        print_failure(f"{device_name} GPU monitoring failed")
        return 0, 0

def test_distributed_inference(primary_host, primary_port):
    """Test inference via primary node"""
    print_header("Running Distributed Inference Test")

    print_info("Primary node", f"{primary_host}:{primary_port}")
    print_info("Model", TEST_MODEL)
    print_info("Prompt", TEST_PROMPT)
    print_info("Max tokens", MAX_TOKENS)

    try:
        print_info("Status", "Sending request...")
        start_time = time.time()

        response = requests.post(
            f"http://{primary_host}:{primary_port}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": TEST_PROMPT}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0
            },
            timeout=180  # 3 minute timeout
        )

        elapsed = time.time() - start_time

        print_info("Status code", response.status_code)
        print_info("Latency", f"{elapsed:.2f}s")

        if response.status_code == 200:
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                generated = result['choices'][0]['message']['content']
                print_success("Inference succeeded")
                print_info("Generated", generated[:300] + "..." if len(generated) > 300 else generated)

                # Show usage stats
                if 'usage' in result:
                    usage = result['usage']
                    print_info("Prompt tokens", usage.get('prompt_tokens', 'N/A'))
                    print_info("Completion tokens", usage.get('completion_tokens', 'N/A'))
                    print_info("Total tokens", usage.get('total_tokens', 'N/A'))

                    # Calculate tokens/sec
                    completion_tokens = usage.get('completion_tokens', 0)
                    if completion_tokens > 0 and elapsed > 0:
                        tokens_per_sec = completion_tokens / elapsed
                        print_info("Tokens/sec", f"{tokens_per_sec:.2f}")

                return True, elapsed
            else:
                print_failure("No choices in response")
                print_info("Response", json.dumps(result, indent=2)[:500])
                return False, elapsed
        else:
            print_failure(f"Request failed: {response.status_code}")
            try:
                error_detail = response.json()
                print_info("Error", json.dumps(error_detail, indent=2)[:500])
            except:
                print_info("Response", response.text[:500])
            return False, elapsed

    except requests.exceptions.Timeout:
        print_failure("Request timed out (180s)")
        return False, 180
    except Exception as e:
        print_failure(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def check_both_gpus_active(thor1_host, thor1_user, thor1_pass, thor2_host, thor2_user, thor2_pass):
    """Check if both GPUs show activity"""
    print_header("Checking GPU Activity on Both Devices")

    thor1_util, thor1_mem = monitor_gpu_usage(thor1_host, thor1_user, thor1_pass, "Thor #1")
    thor2_util, thor2_mem = monitor_gpu_usage(thor2_host, thor2_user, thor2_pass, "Thor #2")

    # Consider GPU active if utilization > 0% OR memory > 500MB
    thor1_active = thor1_util > 0 or thor1_mem > 500
    thor2_active = thor2_util > 0 or thor2_mem > 500

    if thor1_active and thor2_active:
        print_success("Both GPUs show activity")
        return True
    elif thor1_active or thor2_active:
        print_failure("Only one GPU shows activity - may not be distributed")
        return False
    else:
        print_failure("Neither GPU shows significant activity")
        return False

def show_log_grep(host, user, password, device_name, log_file, pattern, context=2):
    """Grep log file and show matches"""
    print_header(f"{device_name} Log - Searching for: {pattern}")

    cmd = f"grep -i -C {context} '{pattern}' {log_file} | tail -50"
    returncode, stdout, stderr = ssh_execute(host, user, password, cmd)

    if returncode == 0 and stdout.strip():
        print(stdout)
        return True
    else:
        print_info("Result", "Pattern not found in log")
        return False

def main():
    """Run full distributed inference test"""
    print("\n" + "=" * 70)
    print("  EXO DISTRIBUTED INFERENCE - TEST 4: FULL DISTRIBUTED")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Pre-flight
    print_header("Pre-Flight Cleanup")
    kill_exo_on_thor(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1")
    kill_exo_on_thor(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2")
    time.sleep(3)

    # Start servers
    pid1, log1 = start_exo_on_thor(
        THOR1_IP, THOR1_USER, THOR1_PASSWORD, THOR1_PORT,
        "Thor #1", "/home/jetson/exo", TEST_MODEL
    )
    if not pid1:
        return 1

    time.sleep(5)

    pid2, log2 = start_exo_on_thor(
        THOR2_IP, THOR2_USER, THOR2_PASSWORD, THOR2_PORT,
        "Thor #2", "/home/thor/exo", TEST_MODEL
    )
    if not pid2:
        kill_exo_on_thor(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1")
        return 1

    try:
        # Wait for servers
        print_header("Waiting for Servers to Initialize")

        ready1 = wait_for_server_ready(THOR1_IP, THOR1_PORT, "Thor #1", timeout=90)
        ready2 = wait_for_server_ready(THOR2_IP, THOR2_PORT, "Thor #2", timeout=90)

        if not (ready1 and ready2):
            print_failure("Servers failed to start")
            return 1

        # Wait for peer discovery and model loading
        print_info("Status", "Waiting 20s for peer discovery and model loading...")
        time.sleep(20)

        # Check GPUs before inference
        print_header("Pre-Inference GPU Check")
        monitor_gpu_usage(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1")
        monitor_gpu_usage(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2")

        # Run inference test
        inference_success, latency = test_distributed_inference(THOR2_IP, THOR2_PORT)

        # Wait a moment for GPU activity to register
        time.sleep(3)

        # Check GPU activity
        both_gpus_active = check_both_gpus_active(
            THOR1_IP, THOR1_USER, THOR1_PASSWORD,
            THOR2_IP, THOR2_USER, THOR2_PASSWORD
        )

        # Search logs for key patterns
        print_header("Log Analysis")

        # Check for peer discovery
        show_log_grep(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1",
                     log1, "peer", context=1)
        show_log_grep(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2",
                     log2, "peer", context=1)

        # Check for errors
        thor1_errors = show_log_grep(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1",
                                     log1, "error", context=2)
        thor2_errors = show_log_grep(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2",
                                     log2, "error", context=2)

        # Results
        results = {
            "Inference Success": inference_success,
            "Both GPUs Active": both_gpus_active,
            "No Errors in Thor #1": not thor1_errors,
            "No Errors in Thor #2": not thor2_errors,
        }

        # Summary
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

        if inference_success:
            print_info("Inference latency", f"{latency:.2f}s")

        if failed == 0:
            print("\n✓ DISTRIBUTED INFERENCE TEST PASSED")
            print("  Successfully generated tokens across 2 Thor devices")
            return 0
        else:
            print(f"\n✗ {failed} TEST(S) FAILED")
            return 1

    finally:
        # Cleanup
        print_header("Cleanup")
        kill_exo_on_thor(THOR1_IP, THOR1_USER, THOR1_PASSWORD, "Thor #1")
        kill_exo_on_thor(THOR2_IP, THOR2_USER, THOR2_PASSWORD, "Thor #2")
        print_success("Cleanup complete")

if __name__ == "__main__":
    sys.exit(main())
