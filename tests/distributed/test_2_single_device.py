#!/usr/bin/env python3
"""
Test 2: Single Device Inference
Tests that exo can run on a single device before attempting distributed setup.
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

def find_exo_processes():
    """Find all running exo processes"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'exo/main.py'],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return [pid for pid in pids if pid]
    except Exception as e:
        print_failure(f"Failed to check for exo processes: {e}")
        return []

def kill_exo_processes():
    """Kill all exo processes"""
    pids = find_exo_processes()
    if pids:
        print_info("Found exo processes", ', '.join(pids))
        for pid in pids:
            try:
                subprocess.run(['kill', '-9', pid], check=True)
                print_success(f"Killed process {pid}")
            except Exception as e:
                print_failure(f"Failed to kill {pid}: {e}")
    else:
        print_info("No exo processes found", "Clean state")

def check_port_available(port):
    """Check if port is available"""
    try:
        result = subprocess.run(
            ['lsof', '-i', f':{port}'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print_failure(f"Port {port} is in use")
            print(result.stdout)
            return False
        else:
            print_success(f"Port {port} available")
            return True
    except Exception as e:
        print_failure(f"Failed to check port {port}: {e}")
        return False

def start_exo_server(model_name="llama-3.2-1b", timeout=60):
    """Start exo server and wait for it to be ready"""
    print_header("Starting Exo Server")

    # Find exo directory
    exo_dir = Path(__file__).parent.parent.parent
    main_py = exo_dir / "exo" / "main.py"

    if not main_py.exists():
        print_failure(f"main.py not found at {main_py}")
        return None

    print_info("Exo directory", exo_dir)
    print_info("Model", model_name)
    print_info("Port", "52415")

    # Set environment variables
    env = os.environ.copy()
    env['DEVICE'] = 'CUDA'
    env['DEBUG'] = '2'

    # Start server
    log_file = Path("/tmp/test_exo_single.log")
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                [sys.executable, str(main_py),
                 '--default-model', model_name,
                 '--inference-engine', 'tinygrad',
                 '--chatgpt-api-port', '52415',
                 '--disable-tui'],
                cwd=str(exo_dir),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group
            )

        print_success(f"Server started with PID {process.pid}")
        print_info("Log file", log_file)

        # Wait for server to be ready
        print_info("Status", "Waiting for server to start...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process is still alive
            if process.poll() is not None:
                print_failure("Server process died")
                with open(log_file) as f:
                    print("\n--- Last 50 lines of log ---")
                    lines = f.readlines()
                    print(''.join(lines[-50:]))
                return None

            # Check if API endpoint is responding
            try:
                response = requests.get("http://localhost:52415/health", timeout=2)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print_success(f"Server ready after {elapsed:.1f}s")
                    return process
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        print_failure(f"Server did not start within {timeout}s")
        return None

    except Exception as e:
        print_failure(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_inference(prompt="Hello, who are you?", max_tokens=20):
    """Test inference via ChatGPT API"""
    print_header("Testing Inference")

    print_info("Prompt", prompt)
    print_info("Max tokens", max_tokens)

    try:
        response = requests.post(
            "http://localhost:52415/v1/chat/completions",
            json={
                "model": "llama-3.2-1b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0
            },
            timeout=120  # 2 minute timeout for inference
        )

        print_info("Status code", response.status_code)

        if response.status_code == 200:
            result = response.json()

            # Extract generated text
            if 'choices' in result and len(result['choices']) > 0:
                generated = result['choices'][0]['message']['content']
                print_success("Inference successful")
                print_info("Generated text", generated[:200] + "..." if len(generated) > 200 else generated)

                # Check usage stats if available
                if 'usage' in result:
                    print_info("Tokens generated", result['usage'].get('completion_tokens', 'N/A'))
                    print_info("Total tokens", result['usage'].get('total_tokens', 'N/A'))

                return True
            else:
                print_failure("No choices in response")
                print_info("Response", json.dumps(result, indent=2))
                return False
        else:
            print_failure(f"Request failed: {response.status_code}")
            print_info("Response", response.text[:500])
            return False

    except requests.exceptions.Timeout:
        print_failure("Inference request timed out (120s)")
        return False
    except Exception as e:
        print_failure(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu_usage():
    """Check if GPU is being used"""
    print_header("Checking GPU Usage")

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            for i, line in enumerate(gpu_info.split('\n')):
                util, mem_used, mem_total = line.split(', ')
                print_info(f"GPU {i}", f"Utilization: {util}%, Memory: {mem_used}/{mem_total} MiB")

                # Check if GPU is being used (>0% utilization or >100MB memory)
                if int(util) > 0 or int(mem_used) > 100:
                    print_success(f"GPU {i} is being used")
                    return True

            print_failure("No significant GPU usage detected")
            print_info("Warning", "Inference may be running on CPU")
            return False
        else:
            print_failure(f"nvidia-smi failed: {result.stderr}")
            return False

    except Exception as e:
        print_failure(f"Failed to check GPU usage: {e}")
        return False

def main():
    """Run single device test"""
    print("\n" + "=" * 70)
    print("  EXO DISTRIBUTED INFERENCE - TEST 2: SINGLE DEVICE")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Pre-flight checks
    print_header("Pre-Flight Checks")
    kill_exo_processes()
    time.sleep(2)  # Wait for cleanup

    if not check_port_available(52415):
        print_failure("Port 52415 not available, exiting")
        return 1

    # Start server
    process = start_exo_server()
    if not process:
        print_failure("Failed to start server")
        return 1

    try:
        # Give server time to initialize
        time.sleep(10)

        # Run tests
        results = {
            "GPU Usage": check_gpu_usage(),
            "Inference Test": test_inference(),
        }

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

        # Check server status before cleanup
        print_header("Final Server Status")
        if process.poll() is None:
            print_success("Server still running")
        else:
            print_failure("Server has stopped")
            print_info("Exit code", process.returncode)

        # Show last log lines
        print("\n--- Last 30 lines of log ---")
        with open("/tmp/test_exo_single.log") as f:
            lines = f.readlines()
            print(''.join(lines[-30:]))

        if failed == 0:
            print("\n✓ SINGLE DEVICE TEST PASSED")
            return 0
        else:
            print(f"\n✗ {failed} TEST(S) FAILED")
            return 1

    finally:
        # Cleanup
        print_header("Cleanup")
        if process and process.poll() is None:
            print_info("Status", "Stopping server...")
            try:
                # Kill entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                print_success("Server stopped cleanly")
            except subprocess.TimeoutExpired:
                print_info("Status", "Force killing server...")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                print_success("Server force killed")
            except Exception as e:
                print_failure(f"Cleanup failed: {e}")

        # Double-check no lingering processes
        time.sleep(2)
        kill_exo_processes()

if __name__ == "__main__":
    sys.exit(main())
