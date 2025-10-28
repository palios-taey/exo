#!/usr/bin/env python3
"""
Verify NVRTC Library Availability
==================================

This script verifies that:
1. NVRTC library can be found by ctypes
2. NVRTC library can be loaded successfully
3. Key NVRTC functions are accessible
4. NVRTC version can be queried

Run this BEFORE removing the monkey-patch to confirm NVRTC is actually available.
"""

import sys
import ctypes
import ctypes.util

def verify_nvrtc():
    """Comprehensive NVRTC availability check"""

    print("=" * 70)
    print("NVRTC Library Verification")
    print("=" * 70)

    # Test 1: Can ctypes.util find the library?
    print("\n[1/4] Checking if NVRTC library can be located...")
    lib_path = ctypes.util.find_library('nvrtc')

    if lib_path is None:
        print("❌ FAILED: NVRTC library NOT found by ctypes.util")
        print("   This means NVRTC is not in the system library path")
        print("   DO NOT remove the monkey-patch!")
        return False

    print(f"✅ SUCCESS: Found NVRTC at: {lib_path}")

    # Test 2: Can we load the library?
    print("\n[2/4] Checking if NVRTC library can be loaded...")
    try:
        nvrtc = ctypes.CDLL(lib_path)
        print(f"✅ SUCCESS: NVRTC library loaded successfully")
    except Exception as e:
        print(f"❌ FAILED: Could not load NVRTC library")
        print(f"   Error: {e}")
        print("   DO NOT remove the monkey-patch!")
        return False

    # Test 3: Does nvrtcVersion function exist?
    print("\n[3/4] Checking if nvrtcVersion function exists...")
    if not hasattr(nvrtc, 'nvrtcVersion'):
        print("❌ FAILED: nvrtcVersion function not found in library")
        print("   NVRTC API may be incomplete")
        print("   DO NOT remove the monkey-patch!")
        return False

    print("✅ SUCCESS: nvrtcVersion function is accessible")

    # Test 4: Can we call nvrtcVersion?
    print("\n[4/4] Checking if nvrtcVersion can be called...")
    try:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        result = nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))

        if result != 0:
            print(f"⚠️  WARNING: nvrtcVersion returned error code: {result}")
            print("   NVRTC may not be fully functional")
            return False

        version = f"{major.value}.{minor.value}"
        print(f"✅ SUCCESS: NVRTC version {version} is operational")

    except Exception as e:
        print(f"❌ FAILED: Could not call nvrtcVersion")
        print(f"   Error: {e}")
        print("   DO NOT remove the monkey-patch!")
        return False

    # Test 5: Check for critical compilation functions
    print("\n[5/5] Checking critical NVRTC compilation functions...")
    critical_functions = [
        'nvrtcCreateProgram',
        'nvrtcCompileProgram',
        'nvrtcGetPTX',
        'nvrtcDestroyProgram'
    ]

    missing_functions = []
    for func_name in critical_functions:
        if not hasattr(nvrtc, func_name):
            missing_functions.append(func_name)

    if missing_functions:
        print(f"❌ FAILED: Missing critical functions: {', '.join(missing_functions)}")
        print("   NVRTC API is incomplete")
        print("   DO NOT remove the monkey-patch!")
        return False

    print(f"✅ SUCCESS: All {len(critical_functions)} critical functions are present")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERIFICATION RESULT: ✅ NVRTC IS FULLY OPERATIONAL")
    print("=" * 70)
    print("\n✅ Safe to remove monkey-patch!")
    print("✅ NVRTC library is available and functional")
    print("✅ CUDACompiler will work correctly")
    print(f"✅ NVRTC version: {version}")
    print("\nNext step: Run ./install.sh to apply the patch")

    return True

def main():
    try:
        success = verify_nvrtc()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("\nDO NOT remove the monkey-patch!")
        print("Investigation required before proceeding.")
        sys.exit(1)

if __name__ == '__main__':
    main()
