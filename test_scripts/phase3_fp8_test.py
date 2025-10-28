#!/usr/bin/env python3
"""
Phase 3: Test FP8 Dtype Support
Tests Agent 3's FP8 dtype mapping fix (F8_E4M3, F8_E5M2)

Usage: python3 phase3_fp8_test.py
"""

import sys

def test_fp8_dtypes_exist():
    """Test that FP8 dtypes are defined in tinygrad"""
    print("\n=== TEST 1: FP8 Dtypes Defined ===")

    try:
        from tinygrad import dtypes

        # Check for FP8 dtype attributes
        has_fp8e4m3 = hasattr(dtypes, 'fp8e4m3')
        has_fp8e5m2 = hasattr(dtypes, 'fp8e5m2')

        print(f"dtypes.fp8e4m3 exists: {has_fp8e4m3}")
        print(f"dtypes.fp8e5m2 exists: {has_fp8e5m2}")

        if has_fp8e4m3 and has_fp8e5m2:
            print("✅ FP8 dtypes defined in tinygrad")
            return True
        else:
            print("❌ FP8 dtypes missing from tinygrad")
            return False

    except Exception as e:
        print(f"❌ FP8 dtype check failed: {e}")
        return False

def test_safe_dtypes_mapping():
    """Test that safe_dtypes includes FP8 mappings"""
    print("\n=== TEST 2: safe_dtypes Mapping ===")

    try:
        from tinygrad.nn.state import safe_dtypes

        print(f"Current safe_dtypes keys: {list(safe_dtypes.keys())}")

        # Check if FP8 dtypes are mapped
        from tinygrad import dtypes

        has_fp8e4m3_mapping = dtypes.fp8e4m3 in safe_dtypes if hasattr(dtypes, 'fp8e4m3') else False
        has_fp8e5m2_mapping = dtypes.fp8e5m2 in safe_dtypes if hasattr(dtypes, 'fp8e5m2') else False

        print(f"fp8e4m3 in safe_dtypes: {has_fp8e4m3_mapping}")
        print(f"fp8e5m2 in safe_dtypes: {has_fp8e5m2_mapping}")

        if has_fp8e4m3_mapping and has_fp8e5m2_mapping:
            print("✅ FP8 dtypes mapped in safe_dtypes")
            return True
        else:
            print("❌ FP8 dtypes missing from safe_dtypes mapping")
            print("⚠️  This is the FIX that Agent 3 needs to apply!")
            return False

    except Exception as e:
        print(f"❌ safe_dtypes check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fp8_tensor_creation():
    """Test creating FP8 tensors"""
    print("\n=== TEST 3: FP8 Tensor Creation ===")

    try:
        from tinygrad import Tensor, dtypes

        # Try to create FP8 tensors
        print("Creating F8_E4M3 tensor...")
        t1 = Tensor([1.0, 2.0, 3.0], dtype=dtypes.fp8e4m3)
        print(f"✅ F8_E4M3 tensor created: {t1}")

        print("Creating F8_E5M2 tensor...")
        t2 = Tensor([1.0, 2.0, 3.0], dtype=dtypes.fp8e5m2)
        print(f"✅ F8_E5M2 tensor created: {t2}")

        print("✅ FP8 tensor creation successful")
        return True

    except KeyError as e:
        print(f"❌ KeyError during tensor creation: {e}")
        print("⚠️  This is the exact error we're trying to fix!")
        return False
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fp8_model_loading():
    """Test loading FP8 model weights (simulated)"""
    print("\n=== TEST 4: FP8 Model Loading (Simulated) ===")

    try:
        from tinygrad.nn.state import safe_load
        import tempfile
        import json

        # Create a minimal safetensors-like structure with FP8 dtype
        # This simulates what happens during Qwen3-FP8 model loading

        print("Simulating FP8 model weight loading...")

        # The actual test would be:
        # safe_load("path/to/qwen3-fp8-model.safetensors")

        # For now, just verify the mapping exists
        from tinygrad.nn.state import safe_dtypes

        try:
            # Try to access F8_E4M3 mapping
            mapping = safe_dtypes.get('F8_E4M3', None)
            if mapping:
                print(f"✅ F8_E4M3 mapping found: {mapping}")
            else:
                print(f"❌ F8_E4M3 mapping not found")
                return False

            mapping = safe_dtypes.get('F8_E5M2', None)
            if mapping:
                print(f"✅ F8_E5M2 mapping found: {mapping}")
            else:
                print(f"❌ F8_E5M2 mapping not found")
                return False

        except Exception as e:
            print(f"❌ Mapping check failed: {e}")
            return False

        print("✅ FP8 model loading test passed (simulated)")
        return True

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Phase 3: FP8 Dtype Testing (Agent 3)")
    print("=" * 60)

    results = {
        "fp8_dtypes_exist": test_fp8_dtypes_exist(),
        "safe_dtypes_mapping": test_safe_dtypes_mapping(),
        "fp8_tensor_creation": test_fp8_tensor_creation(),
        "fp8_model_loading": test_fp8_model_loading(),
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
        print("✅ PHASE 3: ALL TESTS PASSED")
    else:
        print("❌ PHASE 3: SOME TESTS FAILED")
        print("\nExpected Failure: safe_dtypes mapping")
        print("This is the fix Agent 3 needs to implement!")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
