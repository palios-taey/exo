#!/bin/bash
# Installation script for PTX=1 Fix
# Deploys to all nodes: Mira, Thor, Jetson

set -e  # Exit on error

echo "================================================================================"
echo "PTX=1 Fix - Installation Script"
echo "================================================================================"
echo ""
echo "This script will:"
echo "1. Back up inference.py on all nodes"
echo "2. Apply PTX=1 fix (os.environ['PTX'] = '1' at line 5)"
echo "3. Verify modification successful"
echo "4. Deploy to Thor nodes via SSH"
echo ""
echo "================================================================================"
echo ""

# Configuration
SOLUTION_DIR="/home/mira/exo/agents/solutions/agent_1"
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

# Step 1: Verify files exist
echo "Step 1: Verifying files..."
if [ ! -f "$EXO_ROOT/$INFERENCE_FILE" ]; then
    print_error "File not found: $EXO_ROOT/$INFERENCE_FILE"
    exit 1
fi
print_success "inference.py found on Mira"

# Step 2: Create backup on Mira
echo ""
echo "Step 2: Creating backup on Mira..."
BACKUP_FILE="$EXO_ROOT/$INFERENCE_FILE.backup"
if [ -f "$BACKUP_FILE" ]; then
    print_warning "Backup already exists: $BACKUP_FILE"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing backup"
    else
        cp "$EXO_ROOT/$INFERENCE_FILE" "$BACKUP_FILE"
        print_success "Backup created: $BACKUP_FILE"
    fi
else
    cp "$EXO_ROOT/$INFERENCE_FILE" "$BACKUP_FILE"
    print_success "Backup created: $BACKUP_FILE"
fi

# Step 3: Apply fix on Mira
echo ""
echo "Step 3: Applying fix on Mira..."

# Check if already applied
if grep -q "os.environ\['PTX'\] = '1'" "$EXO_ROOT/$INFERENCE_FILE"; then
    print_warning "Fix already applied to Mira"
else
    # Apply fix using Python
    python3 "$SOLUTION_DIR/implementation.py" <<EOF
from implementation import apply_fix_to_file
apply_fix_to_file('$EXO_ROOT/$INFERENCE_FILE', backup=False)
EOF
    if [ $? -eq 0 ]; then
        print_success "Fix applied to Mira"
    else
        print_error "Failed to apply fix to Mira"
        exit 1
    fi
fi

# Step 4: Verify fix on Mira
echo ""
echo "Step 4: Verifying fix on Mira..."
if grep -q "os.environ\['PTX'\] = '1'" "$EXO_ROOT/$INFERENCE_FILE"; then
    # Check line number
    LINE_NUM=$(grep -n "os.environ\['PTX'\] = '1'" "$EXO_ROOT/$INFERENCE_FILE" | head -1 | cut -d: -f1)
    print_success "Fix verified on Mira (line $LINE_NUM)"

    # Check it's before tinygrad imports
    FIRST_TINYGRAD=$(grep -n "from tinygrad\|import tinygrad" "$EXO_ROOT/$INFERENCE_FILE" | head -1 | cut -d: -f1)
    if [ "$LINE_NUM" -lt "$FIRST_TINYGRAD" ]; then
        print_success "PTX=1 set BEFORE tinygrad imports (correct)"
    else
        print_error "PTX=1 at line $LINE_NUM is AFTER tinygrad import at line $FIRST_TINYGRAD"
        print_error "This will NOT work! Manual fix required."
        exit 1
    fi
else
    print_error "Fix NOT found in Mira file"
    exit 1
fi

# Step 5: Deploy to Thor #1
echo ""
echo "Step 5: Deploying to Thor #1 (10.0.0.78)..."

# Create backup on Thor #1
sshpass -p "$THOR_1_PASS" ssh "$THOR_1_HOST" "cp $THOR_1_PATH/$INFERENCE_FILE $THOR_1_PATH/$INFERENCE_FILE.backup" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Backup created on Thor #1"
else
    print_warning "Backup failed on Thor #1 (may not exist yet)"
fi

# Copy modified file to Thor #1
sshpass -p "$THOR_1_PASS" scp "$EXO_ROOT/$INFERENCE_FILE" "$THOR_1_HOST:$THOR_1_PATH/$INFERENCE_FILE"
if [ $? -eq 0 ]; then
    print_success "File copied to Thor #1"
else
    print_error "Failed to copy to Thor #1"
    exit 1
fi

# Verify on Thor #1
THOR_1_CHECK=$(sshpass -p "$THOR_1_PASS" ssh "$THOR_1_HOST" "grep -c \"os.environ\['PTX'\] = '1'\" $THOR_1_PATH/$INFERENCE_FILE" 2>/dev/null)
if [ "$THOR_1_CHECK" = "1" ]; then
    print_success "Fix verified on Thor #1"
else
    print_error "Fix NOT found on Thor #1"
    exit 1
fi

# Step 6: Deploy to Jetson
echo ""
echo "Step 6: Deploying to Jetson (10.0.0.93)..."

# Create backup on Jetson
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" "cp $JETSON_PATH/$INFERENCE_FILE $JETSON_PATH/$INFERENCE_FILE.backup" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Backup created on Jetson"
else
    print_warning "Backup failed on Jetson (may not exist yet)"
fi

# Copy modified file to Jetson
sshpass -p "$JETSON_PASS" scp "$EXO_ROOT/$INFERENCE_FILE" "$JETSON_HOST:$JETSON_PATH/$INFERENCE_FILE"
if [ $? -eq 0 ]; then
    print_success "File copied to Jetson"
else
    print_error "Failed to copy to Jetson"
    exit 1
fi

# Verify on Jetson
JETSON_CHECK=$(sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" "grep -c \"os.environ\['PTX'\] = '1'\" $JETSON_PATH/$INFERENCE_FILE" 2>/dev/null)
if [ "$JETSON_CHECK" = "1" ]; then
    print_success "Fix verified on Jetson"
else
    print_error "Fix NOT found on Jetson"
    exit 1
fi

# Final summary
echo ""
echo "================================================================================"
echo "Installation Complete!"
echo "================================================================================"
echo ""
print_success "Fix applied to all nodes:"
echo "  - Mira:   $EXO_ROOT/$INFERENCE_FILE"
echo "  - Thor:   $THOR_1_PATH/$INFERENCE_FILE"
echo "  - Jetson: $JETSON_PATH/$INFERENCE_FILE"
echo ""
print_success "Backups created:"
echo "  - Mira:   $BACKUP_FILE"
echo "  - Thor:   $THOR_1_PATH/$INFERENCE_FILE.backup"
echo "  - Jetson: $JETSON_PATH/$INFERENCE_FILE.backup"
echo ""
echo "Next steps:"
echo "1. Run test.sh to verify the fix works"
echo "2. Start exo servers on Thor nodes"
echo "3. Monitor logs for PTXRenderer confirmation"
echo ""
echo "If issues occur:"
echo "  Run: ./rollback.sh"
echo ""
echo "================================================================================"
