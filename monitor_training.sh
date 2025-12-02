#!/bin/bash
# Monitor DQN training progress

echo "🎮 Tetris DQN Training Monitor"
echo "================================"
echo ""

# Check if training is running
if ps aux | grep "src.train" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Show resource usage
    echo "📊 Resource Usage:"
    ps aux | grep "src.train" | grep -v grep | awk '{print "   CPU: " $3 "%, Memory: " $4 "%"}'
    echo ""
    
    # Show checkpoints
    echo "💾 Saved Checkpoints:"
    if [ -d "models/checkpoints" ]; then
        ls -lth models/checkpoints/*.pt 2>/dev/null | head -10 | awk '{print "   " $9 " (" $5 ")"}'
        echo ""
        echo "📈 Total checkpoints: $(ls models/checkpoints/*.pt 2>/dev/null | wc -l)"
    else
        echo "   No checkpoints yet"
    fi
    echo ""
    
    # Show plots
    if [ -f "plots/reward_curve.png" ]; then
        echo "📊 Learning curve: plots/reward_curve.png"
    fi
    
else
    echo "❌ Training is NOT running"
    echo ""
    echo "💾 Final Checkpoints:"
    if [ -d "models/checkpoints" ]; then
        ls -lth models/checkpoints/*.pt 2>/dev/null | head -10 | awk '{print "   " $9 " (" $5 ")"}'
    fi
fi

echo ""
echo "================================"

