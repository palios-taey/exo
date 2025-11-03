#!/bin/bash
# Automated Git Deployment to Thor #1 and Thor #2
# Usage: ./deploy_to_thors.sh "commit message"

set -e  # Exit on any error

COMMIT_MSG="${1:-Update from Mira}"
BRANCH="thor-compatibility-2025-11-02"
FORK_REMOTE="fork"

THOR1_USER="jetson"
THOR1_HOST="10.0.0.93"
THOR1_PATH="/home/jetson/exo-clean"

THOR2_USER="thor"
THOR2_HOST="10.0.0.78"
THOR2_PATH="/home/thor/exo-clean"

GITHUB_REPO="git@github.com:palios-taey/exo.git"

LOG_FILE="/tmp/deploy_$(date +%Y%m%d_%H%M%S).log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[DEPLOY]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# Step 1: Verify we're on Mira in correct directory
log "Step 1: Verifying environment..."
if [[ ! -d ".git" ]]; then
    error "Not in a git repository. Run from /home/mira/exo"
fi

CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    warn "Currently on branch '$CURRENT_BRANCH', expected '$BRANCH'"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        error "Deployment cancelled"
    fi
fi

# Step 2: Check for uncommitted changes, stage and commit
log "Step 2: Checking for changes to commit..."
if git diff-index --quiet HEAD --; then
    log "No changes to commit, using existing HEAD"
else
    log "Staging all changes..."
    git add -A

    log "Committing with message: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
fi

MIRA_COMMIT=$(git rev-parse HEAD)
log "Mira commit: $MIRA_COMMIT"

# Step 3: Push to GitHub
log "Step 3: Pushing to GitHub ($FORK_REMOTE/$BRANCH)..."
git push "$FORK_REMOTE" "$BRANCH" || error "Failed to push to GitHub"

# Step 4: Initialize or update Thor #1
log "Step 4: Deploying to Thor #1 ($THOR1_HOST)..."

ssh "${THOR1_USER}@${THOR1_HOST}" bash <<'EOF'
set -e

THOR1_PATH="/home/jetson/exo-clean"
BRANCH="thor-compatibility-2025-11-02"
GITHUB_REPO="git@github.com:palios-taey/exo.git"

if [[ -d "$THOR1_PATH/.git" ]]; then
    echo "Git repo exists, pulling latest..."
    cd "$THOR1_PATH"

    # Check for uncommitted changes (enforces read-only policy)
    if ! git diff-index --quiet HEAD --; then
        echo "ERROR: Uncommitted changes detected on Thor #1"
        echo "Thors should be read-only deployment targets"
        exit 1
    fi

    git fetch origin
    git reset --hard origin/$BRANCH
else
    echo "No git repo, initializing from GitHub..."

    # Backup existing directory if present
    if [[ -d "$THOR1_PATH" ]]; then
        mv "$THOR1_PATH" "${THOR1_PATH}.backup.$(date +%s)"
    fi

    git clone -b "$BRANCH" "$GITHUB_REPO" "$THOR1_PATH"
    cd "$THOR1_PATH"
fi

echo "Thor #1 commit: $(git rev-parse HEAD)"
EOF

if [[ $? -ne 0 ]]; then
    error "Thor #1 deployment failed"
fi

# Step 5: Initialize or update Thor #2
log "Step 5: Deploying to Thor #2 ($THOR2_HOST)..."

ssh "${THOR2_USER}@${THOR2_HOST}" bash <<'EOF'
set -e

THOR2_PATH="/home/thor/exo-clean"
BRANCH="thor-compatibility-2025-11-02"
GITHUB_REPO="git@github.com:palios-taey/exo.git"

if [[ -d "$THOR2_PATH/.git" ]]; then
    echo "Git repo exists, pulling latest..."
    cd "$THOR2_PATH"

    if ! git diff-index --quiet HEAD --; then
        echo "ERROR: Uncommitted changes detected on Thor #2"
        exit 1
    fi

    git fetch origin
    git reset --hard origin/$BRANCH
else
    echo "No git repo, initializing from GitHub..."

    if [[ -d "$THOR2_PATH" ]]; then
        mv "$THOR2_PATH" "${THOR2_PATH}.backup.$(date +%s)"
    fi

    git clone -b "$BRANCH" "$GITHUB_REPO" "$THOR2_PATH"
    cd "$THOR2_PATH"
fi

echo "Thor #2 commit: $(git rev-parse HEAD)"
EOF

if [[ $? -ne 0 ]]; then
    error "Thor #2 deployment failed"
fi

# Step 6: Verify all 3 nodes have identical commit hash
log "Step 6: Verifying synchronization..."

THOR1_COMMIT=$(ssh "${THOR1_USER}@${THOR1_HOST}" "cd $THOR1_PATH && git rev-parse HEAD")
THOR2_COMMIT=$(ssh "${THOR2_USER}@${THOR2_HOST}" "cd $THOR2_PATH && git rev-parse HEAD")

log "Mira:    $MIRA_COMMIT"
log "Thor #1: $THOR1_COMMIT"
log "Thor #2: $THOR2_COMMIT"

if [[ "$MIRA_COMMIT" != "$THOR1_COMMIT" ]] || [[ "$MIRA_COMMIT" != "$THOR2_COMMIT" ]]; then
    error "Commit hash mismatch! Deployment FAILED"
fi

log "✅ SUCCESS: All 3 nodes synchronized at commit $MIRA_COMMIT"
log "Deployment log: $LOG_FILE"
