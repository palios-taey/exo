#!/usr/bin/env python3
"""
Direct File Edit: Remove NVRTC Monkey-Patch
===========================================

This script directly removes lines 2-36 from inference.py.
Alternative to using patch command (which can be finicky with format).
"""

import sys
from pathlib import Path

def remove_monkeypatch(file_path):
    """Remove NVRTC monkey-patch from inference.py"""

    print(f"Processing: {file_path}")

    # Read original file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    print(f"Original file: {len(lines)} lines")

    # Check if monkey-patch is present
    if len(lines) < 36 or 'NVRTC MONKEY-PATCH' not in lines[1]:
        print("ERROR: Monkey-patch not found or already removed!")
        print(f"Line 2 content: {lines[1] if len(lines) > 1 else 'N/A'}")
        return False

    # Verify structure
    if '_apply_nvrtc_patch()' not in lines[35]:
        print("ERROR: Expected structure not found at line 36!")
        print(f"Line 36 content: {lines[35]}")
        return False

    print("✅ Monkey-patch detected at lines 2-36")

    # Create backup
    backup_path = f"{file_path}.nvrtc_backup"
    print(f"Creating backup: {backup_path}")

    with open(backup_path, 'w') as f:
        f.writelines(lines)

    print("✅ Backup created")

    # Remove lines 2-36 (indices 1-35 inclusive, zero-based)
    # Keep line 1 (index 0), skip lines 2-36, keep rest
    new_lines = [lines[0]] + lines[36:]

    print(f"New file will have: {len(new_lines)} lines (removed {len(lines) - len(new_lines)} lines)")

    # Write modified file
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

    print("✅ File modified")

    # Verify
    with open(file_path, 'r') as f:
        verify_lines = f.readlines()

    if 'NVRTC MONKEY-PATCH' in ''.join(verify_lines[:50]):
        print("❌ ERROR: Monkey-patch still present after modification!")
        # Restore backup
        with open(backup_path, 'r') as f:
            original = f.readlines()
        with open(file_path, 'w') as f:
            f.writelines(original)
        print("Restored from backup")
        return False

    print("✅ Verification passed: Monkey-patch removed")
    print(f"✅ SUCCESS: {file_path} modified")
    print(f"   Backup saved at: {backup_path}")

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 apply_fix.py /path/to/inference.py")
        print("\nExample:")
        print("  python3 apply_fix.py /home/thor/exo/exo/inference/tinygrad/inference.py")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    if not file_path.name == 'inference.py':
        print(f"WARNING: Expected filename 'inference.py', got '{file_path.name}'")
        print("Continue anyway? (yes/no): ", end='')
        if input().lower() != 'yes':
            print("Aborted")
            sys.exit(1)

    success = remove_monkeypatch(file_path)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
