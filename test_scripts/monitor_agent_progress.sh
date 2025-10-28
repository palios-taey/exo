#!/bin/bash
# Monitor progress of Agents 1-4 to know when to start testing
# Usage: bash monitor_agent_progress.sh

REPORTS_DIR="/home/mira/exo/agent_reports"

echo "============================================================"
echo "Exo Fix Mission - Agent Progress Monitor"
echo "============================================================"
echo ""

check_agent_status() {
    local agent_num=$1
    local agent_name=$2
    local report_file="${REPORTS_DIR}/AGENT${agent_num}_*.md"

    if ls $report_file 1> /dev/null 2>&1; then
        # Report exists, check status
        if grep -q "Status.*COMPLETED" $report_file; then
            echo "✅ Agent $agent_num ($agent_name): COMPLETED"
            return 0
        elif grep -q "Status.*BLOCKED" $report_file; then
            echo "⚠️  Agent $agent_num ($agent_name): BLOCKED"
            return 2
        elif grep -q "Status.*IN_PROGRESS" $report_file; then
            echo "🔄 Agent $agent_num ($agent_name): IN_PROGRESS"
            return 1
        else
            echo "❓ Agent $agent_num ($agent_name): UNKNOWN"
            return 3
        fi
    else
        echo "⏳ Agent $agent_num ($agent_name): NOT STARTED"
        return 4
    fi
}

while true; do
    clear
    echo "============================================================"
    echo "Exo Fix Mission - Agent Progress Monitor"
    echo "Updated: $(date)"
    echo "============================================================"
    echo ""

    check_agent_status 1 "Custom Tinygrad Builder"
    agent1_status=$?

    check_agent_status 2 "NVPTXCompiler Integrator"
    agent2_status=$?

    check_agent_status 3 "FP8 Dtype Fixer"
    agent3_status=$?

    check_agent_status 4 "Discovery Coordinator"
    agent4_status=$?

    check_agent_status 5 "Hardware Tester (ME)"
    agent5_status=$?

    echo ""
    echo "============================================================"
    echo "Testing Readiness"
    echo "============================================================"
    echo ""

    if [ $agent1_status -eq 0 ]; then
        echo "✅ Phase 1 testing: READY"
        echo "   Run: ssh thor@10.0.0.78 'python3 /home/mira/exo/test_scripts/phase1_tinygrad_test.py'"
    else
        echo "⏳ Phase 1 testing: WAITING for Agent 1"
    fi

    if [ $agent2_status -eq 0 ]; then
        echo "✅ Phase 2 testing: READY"
        echo "   Run: ssh thor@10.0.0.78 'python3 /home/mira/exo/test_scripts/phase2_nvptx_test.py'"
    else
        echo "⏳ Phase 2 testing: WAITING for Agent 2"
    fi

    if [ $agent3_status -eq 0 ]; then
        echo "✅ Phase 3 testing: READY"
        echo "   Run: ssh thor@10.0.0.78 'python3 /home/mira/exo/test_scripts/phase3_fp8_test.py'"
    else
        echo "⏳ Phase 3 testing: WAITING for Agent 3"
    fi

    if [ $agent4_status -eq 0 ]; then
        echo "✅ Phase 4 testing: READY"
        echo "   Run: ssh thor@10.0.0.78 'python3 /home/mira/exo/test_scripts/phase4_discovery_test.py'"
    else
        echo "⏳ Phase 4 testing: WAITING for Agent 4"
    fi

    if [ $agent1_status -eq 0 ] && [ $agent2_status -eq 0 ] && [ $agent3_status -eq 0 ] && [ $agent4_status -eq 0 ]; then
        echo ""
        echo "🎯 ALL AGENTS COMPLETE!"
        echo "✅ Phase 5 testing: READY"
        echo "   Run: bash /home/mira/exo/test_scripts/phase5_end_to_end_test.sh"
        echo ""
        echo "============================================================"
        break
    fi

    echo ""
    echo "Refreshing in 30 seconds... (Ctrl+C to exit)"
    sleep 30
done

echo ""
echo "Ready to begin hardware testing!"
