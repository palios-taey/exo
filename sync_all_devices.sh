#!/bin/bash
# sync_all_devices.sh - Sync exo code to both Thor devices
# Usage: ./sync_all_devices.sh [branch-name]
# Default branch: claude-cuda13-blackwell-patches

set -e  # Exit on error

BRANCH=${1:-claude-cuda13-blackwell-patches}
MIRA_DIR="/home/mira/exo"

echo "=== EXO SYNC TO BOTH THOR DEVICES ==="
echo "Branch: $BRANCH"
echo ""

# Check Mira git state
cd $MIRA_DIR
if ! git diff-index --quiet HEAD --; then
    echo "ERROR: Mira has uncommitted changes. Commit or stash first."
    git status
    exit 1
fi

# Push to fork
echo "[Mira] Pushing $BRANCH to fork..."
git push fork $BRANCH

# Sync Thor #1 (10.0.0.78)
echo ""
echo "[Thor #1] Syncing to 10.0.0.78..."
ssh thor@10.0.0.78 "cd /home/thor/exo && \
  git fetch fork && \
  git checkout $BRANCH && \
  git reset --hard fork/$BRANCH && \
  echo 'Thor #1 synced to:' && \
  git log --oneline -1"

# Sync Jetson #2 (10.0.0.93)
echo ""
echo "[Jetson #2] Syncing to 10.0.0.93..."
ssh jetson@10.0.0.93 "cd /home/jetson/exo && \
  git fetch fork && \
  git checkout $BRANCH && \
  git reset --hard fork/$BRANCH && \
  echo 'Jetson #2 synced to:' && \
  git log --oneline -1"

echo ""
echo "=== SYNC COMPLETE ==="
echo "All 3 machines now at commit:"
git log --oneline -1
