#!/bin/bash

echo "🔍 Finding benchmark processes..."
ps aux | grep -E "click_main|kmmlu_main|haerae_main|hrm8k_main|kobalt_main" | grep -v grep

echo ""
echo "🛑 Killing all benchmark processes..."
pkill -9 -f "benchmarks/"

echo ""
echo "✅ Done! Verifying..."
REMAINING=$(ps aux | grep -E "click_main|kmmlu_main|haerae_main|hrm8k_main|kobalt_main" | grep -v grep | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ All benchmark processes terminated successfully"
else
    echo "⚠️  Warning: $REMAINING processes still running"
    ps aux | grep -E "click_main|kmmlu_main|haerae_main|hrm8k_main|kobalt_main" | grep -v grep
fi
