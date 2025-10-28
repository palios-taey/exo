#!/bin/bash
# Rollback script for PTX=1 Fix
# Restores original inference.py on all nodes

set -e  # Exit on error

echo "================================================================================"
echo "PTX=1 Fix - Rollback Script"
echo "================================================================================"
echo ""
echo "⚠️  WARNING: This will restore original inference.py (without PTX=1 fix)"
echo ""
read -p "Continue with rollback? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled"
    exit 0
fi
echo ""

# Configuration
EXO_ROOT="/home/mira/exo"
INFERENCE_FILE="exo/inference/tinygrad/inference.py"

THOR_1_HOST="thor@10.0.0.78"
THOR_1_PATH="/home/thor/exo"
THOR_1_PASS="papaDons1001s$"

JETSON_HOST="jetson@10.0.0.93"
JETSON_PATH="/home/jetson/exo"
JETSON_PASS="papaDons1001s$"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Rollback Mira
echo "Step 1: Rolling back Mira..."
MIRA_BACKUP="$EXO_ROOT/$INFERENCE_FILE.backup"
if [ -f "$MIRA_BACKUP" ]; then
    cp "$MIRA_BACKUP" "$EXO_ROOT/$INFERENCE_FILE"
    print_success "Mira: Restored from backup"

    # Verify PTX=1 removed
    if grep -q "os.environ\['PTX'\] = '1'" "$EXO_ROOT/$INFERENCE_FILE"; then
        print_warning "Mira: PTX=1 still present after rollback"
    else
        print_success "Mira: PTX=1 removed"
    fi
else
    print_error "Mira: Backup not found at $MIRA_BACKUP"
    exit 1
fi

# Step 2: Rollback Thor #1
echo ""
echo "Step 2: Rolling back Thor #1..."
THOR_1_RESULT=$(sshpass -p "$THOR_1_PASS" ssh "$THOR_1_HOST" "
if [ -f $THOR_1_PATH/$INFERENCE_FILE.backup ]; then
    cp $THOR_1_PATH/$INFERENCE_FILE.backup $THOR_1_PATH/$INFERENCE_FILE
    echo 'SUCCESS'
else
    echo 'NO_BACKUP'
fi
" 2>/dev/null)

if [ "$THOR_1_RESULT" = "SUCCESS" ]; then
    print_success "Thor #1: Restored from backup"

    # Verify PTX=1 removed
    THOR_1_CHECK=$(sshpass -p "$THOR_1_PASS" ssh "$THOR_1_HOST" "grep -c \"os.environ\['PTX'\] = '1'\" $THOR_1_PATH/$INFERENCE_FILE" 2>/dev/null || echo "0")
    if [ "$THOR_1_CHECK" = "0" ]; then
        print_success "Thor #1: PTX=1 removed"
    else
        print_warning "Thor #1: PTX=1 still present"
    fi
elif [ "$THOR_1_RESULT" = "NO_BACKUP" ]; then
    print_error "Thor #1: Backup not found"
    exit 1
else
    print_error "Thor #1: Rollback failed"
    exit 1
fi

# Step 3: Rollback Jetson
echo ""
echo "Step 3: Rolling back Jetson..."
JETSON_RESULT=$(sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" "
if [ -f $JETSON_PATH/$INFERENCE_FILE.backup ]; then
    cp $JETSON_PATH/$INFERENCE_FILE.backup $JETSON_PATH/$INFERENCE_FILE
    echo 'SUCCESS'
else
    echo 'NO_BACKUP'
fi
" 2>/dev/null)

if [ "$JETSON_RESULT" = "SUCCESS" ]; then
    print_success "Jetson: Restored from backup"

    # Verify PTX=1 removed
    JETSON_CHECK=$(sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" "grep -c \"os.environ\['PTX'\] = '1'\" $JETSON_PATH/$INFERENCE_FILE" 2>/dev/null || echo "0")
    if [ "$JETSON_CHECK" = "0" ]; then
        print_success "Jetson: PTX=1 removed"
    else
        print_warning "Jetson: PTX=1 still present"
    fi
elif [ "$JETSON_RESULT" = "NO_BACKUP" ]; then
    print_error "Jetson: Backup not found"
    exit 1
else
    print_error "Jetson: Rollback failed"
    exit 1
fi

# Step 4: Stop running servers (optional)
echo ""
echo "Step 4: Stopping exo servers (if running)..."
echo ""
read -p "Stop exo servers on Thor nodes? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping Thor #1..."
    sshpass -p "$THOR_1_PASS" ssh "$THOR_1_HOST" "pkill -f 'python3.*exo/main.py'" 2>/dev/null || echo "(No process found)"

    echo "Stopping Jetson..."
    sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" "pkill -f 'python3.*exo/main.py'" 2>/dev/null || echo "(No process found)"

    print_success "Exo servers stopped"
fi

# Final summary
echo ""
echo "================================================================================"
echo "Rollback Complete!"
echo "================================================================================"
echo ""
print_success "Original inference.py restored on all nodes:"
echo "  - Mira:   $EXO_ROOT/$INFERENCE_FILE"
echo "  - Thor:   $THOR_1_PATH/$INFERENCE_FILE"
echo "  - Jetson: $JETSON_PATH/$INFERENCE_FILE"
echo ""
print_warning "PTX=1 fix removed - system back to original state"
echo ""
echo "Backups preserved at:"
echo "  - Mira:   $MIRA_BACKUP"
echo "  - Thor:   $THOR_1_PATH/$INFERENCE_FILE.backup"
echo "  - Jetson: $JETSON_PATH/$INFERENCE_FILE.backup"
echo ""
echo "To re-apply the fix:"
echo "  Run: ./install.sh"
echo ""
echo "================================================================================"
