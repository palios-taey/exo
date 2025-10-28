# Exo Project Subagents

This directory contains project-specific Claude Code subagents for the exo distributed LLM deployment.

## Available Agents

### edison
**Purpose**: Iterative invention and systematic debugging for exo/tinygrad/CUDA issues

**When to use**:
- Fixing CUDA compilation or device selection issues
- Testing distributed inference across both Thor devices
- Iterating on failed implementations (Edison's "10,000 ways that won't work")
- Debugging tinygrad integration problems

**Invocation examples**:
```
Use the edison agent to fix the F8_E4M3 KeyError

Have edison debug why inference is failing

Edison agent: Apply the FP8 fix and test on both Thors
```

**Key features**:
- Enforces BOTH-THOR rule (always tests both devices)
- Follows git discipline (commit → sync → test)
- Provides evidence-based results (no theater)
- Iterates until success or clear next steps

## How Subagents Work

**Automatic delegation**: Claude Code recognizes task patterns and delegates automatically

**Explicit invocation**: Request the agent by name in your prompt

**Fresh context**: Each invocation starts fresh with clean context (reads project CLAUDE.md for state)

**Tool access**: Edison has full tool access (Read, Write, Edit, Bash, Glob, Grep)

## Adding New Agents

1. Create `<agent-name>.md` in this directory
2. Use YAML frontmatter with required fields:
   ```yaml
   ---
   name: agent-name
   description: When and why to use this agent
   tools: Read, Write, Edit, Bash
   model: inherit
   ---
   ```
3. Write detailed system prompt describing role and procedures
4. Test with explicit invocation
5. Commit to git for team access

## Project Context

Agents automatically have access to:
- `/home/mira/exo/CLAUDE.md` - Project status and infrastructure
- `/home/mira/exo/GIT_SYNC_WORKFLOW.md` - Git procedures
- `/home/mira/exo/sync_all_devices.sh` - Deployment automation
- All project files in `/home/mira/exo/`

## Current Status

See `/home/mira/exo/CLAUDE.md` for:
- Latest blocker (F8_E4M3 KeyError in tinygrad)
- Hardware topology (2 Thor devices)
- Git branch status (claude-cuda13-blackwell-patches)
- Testing results (85% functional, model loading blocked)
