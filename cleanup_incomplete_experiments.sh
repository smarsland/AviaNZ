#!/bin/bash
# ============================================================
# Cleanup script for incomplete experiments
#
# Problem: If experiments halt mid-run, some experiment folders
#          are created but don't have a result.json file.
#          This prevents re-running since the script thinks
#          those experiments already exist.
#
# Solution: Delete experiment folders without result.json
#
# Usage:
#   ./cleanup_incomplete_experiments.sh [EXPERIMENTS_DIR]
#
# If no directory specified, uses default:
#   /local/scratch/$USER/experiments_matched
# ============================================================

set -e

# Default path (matches run_matched_experiments.sh)
DEFAULT_EXPERIMENTS_DIR="/local/scratch/$USER/experiments_matched"
EXPERIMENTS_DIR="${1:-$DEFAULT_EXPERIMENTS_DIR}"

# Check if directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Error: Directory does not exist: $EXPERIMENTS_DIR"
    echo ""
    echo "Usage: $0 [EXPERIMENTS_DIR]"
    echo "  Default: $DEFAULT_EXPERIMENTS_DIR"
    exit 1
fi

echo "============================================================"
echo " Cleanup Incomplete Experiments"
echo "============================================================"
echo "  Directory: $EXPERIMENTS_DIR"
echo "============================================================"
echo ""

# Remove main all_results.json if it exists (will be regenerated)
ALL_RESULTS="$EXPERIMENTS_DIR/all_results.json"
if [ -f "$ALL_RESULTS" ]; then
    echo "🗑️  Removing main results file: $ALL_RESULTS"
    rm -f "$ALL_RESULTS"
fi

# Count complete and incomplete experiments
INCOMPLETE=0
COMPLETE=0

# Find all subdirectories in experiments folder
echo "Scanning for incomplete experiments..."
echo ""

for exp_dir in "$EXPERIMENTS_DIR"/*/ ; do
    # Skip if no subdirectories found
    [ -d "$exp_dir" ] || continue
    
    exp_name=$(basename "$exp_dir")
    result_file="$exp_dir/result.json"
    
    # Check if result.json exists
    if [ ! -f "$result_file" ]; then
        echo "❌ INCOMPLETE: $exp_name"
        echo "   → Removing: $exp_dir"
        rm -rf "$exp_dir"
        ((INCOMPLETE++))
    else
        echo "✓ Complete: $exp_name"
        ((COMPLETE++))
    fi
done

echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
echo "  Complete experiments   : $COMPLETE (kept)"
echo "  Incomplete experiments : $INCOMPLETE (removed)"
echo "============================================================"
echo ""

if [ $INCOMPLETE -gt 0 ]; then
    echo "✅ Cleaned up $INCOMPLETE incomplete experiment(s)"
    echo "   You can now rerun run_matched_experiments.sh"
else
    echo "✅ No cleanup needed - all experiments are complete!"
fi

echo ""
