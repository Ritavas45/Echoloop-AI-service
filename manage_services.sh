#!/bin/bash

# ============================================================================
# Echoloop AI - Native macOS launchd Service Control Script
# ============================================================================

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

API_PLIST="/Users/ritavas/Library/LaunchAgents/com.echoloop.api.plist"

check_service() {
    local label=$1
    launchctl list | grep "$label" >/dev/null
    return $?
}

print_status() {
    echo "--------------------------------------------------"
    echo " Echoloop AI Service Status (Unified Backend)"
    echo "--------------------------------------------------"
    
    if check_service "com.echoloop.api"; then
        local pid=$(launchctl list | grep "com.echoloop.api" | awk '{print $1}')
        local status=$(launchctl list | grep "com.echoloop.api" | awk '{print $2}')
        if [ "$pid" = "-" ] || [ -z "$pid" ]; then
            echo "● Unified Backend (com.echoloop.api): Stopped (Code: $status)"
        else
            echo "● Unified Backend (com.echoloop.api): Running (PID: $pid)"
            echo "  (Includes FastAPI Web Server & background Orchestrator scheduler)"
        fi
    else
        echo "○ Unified Backend (com.echoloop.api): Not Loaded"
    fi
    echo "--------------------------------------------------"
}

case "$1" in
    start)
        echo "Loading and starting Echoloop AI Unified Backend..."
        
        # Unload legacy orchestrator plist if it exists
        old_orch_plist="/Users/ritavas/Library/LaunchAgents/com.echoloop.orchestrator.plist"
        if [ -f "$old_orch_plist" ]; then
            launchctl unload "$old_orch_plist" 2>/dev/null || launchctl bootout gui/501 "$old_orch_plist" 2>/dev/null
            rm -f "$old_orch_plist"
            echo "Removed legacy standalone orchestrator plist."
        fi

        # Unload first just in case
        launchctl unload "$API_PLIST" 2>/dev/null || launchctl bootout gui/501 "$API_PLIST" 2>/dev/null
        
        # Load
        launchctl load "$API_PLIST" 2>/dev/null || launchctl bootstrap gui/501 "$API_PLIST"
        
        sleep 2
        print_status
        ;;
    stop)
        echo "Stopping and unloading Echoloop AI Unified Backend..."
        launchctl unload "$API_PLIST" 2>/dev/null || launchctl bootout gui/501 "$API_PLIST" 2>/dev/null
        echo "Services stopped."
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    status)
        print_status
        ;;
    logs)
        echo "Tailing logs (Press Ctrl+C to exit)..."
        tail -n 50 -f "$DIR/logs/api_launchd_stderr.log" "$DIR/logs/api_launchd_stdout.log"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
