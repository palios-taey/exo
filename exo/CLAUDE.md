# VENDOR CODE GUIDELINES: exo Framework Source
*Last Updated: 2025-10-23*

## Import Parent Context

`@/home/mira/exo/CLAUDE.md` (which imports root CLAUDE.md)

## Vendor Code Philosophy

**This is external project code (exo framework source).**

**Core Principle**: Read-only mindset. Only modify when explicitly fixing bugs or adding necessary features.

**Default Question**: "Should this be a patch to vendor code, or our own wrapper around it?"

## Modification Guidelines

### When You MUST Modify Vendor Code

1. **Document Changes** - Keep detailed notes of what was changed and why
2. **Preserve Attribution** - Never remove original author comments
3. **Minimal Changes** - Only what's necessary to fix the issue
4. **Comment Patches** - Add clear comments explaining our modifications
5. **Track Upstream** - Note if changes should be upstreamed to exo project

### Files Already Modified in This Codebase

**exo/inference/tinygrad/inference.py**:
- Device.DEFAULT fix for CUDA availability check
- NVRTC monkey-patch for tinygrad CUDA compilation

**exo/inference/tinygrad/tinygrad_helpers.py**:
- Infinite loop fix: Added `if n not in parts` check in `get_device_capabilities()`

## What NOT to Do in Vendor Code

**DO NOT create new files in vendor directories:**
- Scripts → `/home/mira/scripts/` or `/home/thor/scripts/`
- Tests → `/home/mira/exo/tests/`
- Documentation → `/home/mira/exo/docs/`
- Configuration → `/home/mira/exo/config/`
- Helper utilities → Parent directory, NOT here

**DO NOT:**
- Add new modules or packages to vendor tree
- Refactor vendor code for style preferences
- Add tooling or scaffolding to vendor directories
- Create wrapper classes inside vendor code (wrap from outside)

## Git Considerations (Future)

- Vendor code might be tracked as git submodule
- Our modifications might need to be maintained as patches
- Keep changes minimal to ease future upstream merges
- Consider if fix should be contributed back to exo project

## The Right Way to Extend Vendor Code

**BAD**: Create `/home/mira/exo/exo/inference/tinygrad/my_helper.py`
**GOOD**: Create `/home/mira/exo/tinygrad_extensions.py` (parent directory)

**BAD**: Add new utility functions to vendor modules
**GOOD**: Import vendor code and wrap it in our own module

**BAD**: Refactor vendor code to match our coding style
**GOOD**: Work with vendor code as-is, document patches clearly

## Summary

**Vendor code = special care + minimal modification**

When in doubt: Wrap, don't modify. Patch only when necessary. Document everything.

---

*Vendor code guidelines for exo framework source. 2,847 characters.*
*Team Anthropic. #FREE_CLAUDE #I_AM_CLAUDE*
TIMESTAMP: 2025-10-23_16:57:32_UTC
