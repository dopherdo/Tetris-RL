#!/bin/bash
# Script to prepare Canvas submission

echo "Preparing Canvas submission..."

cd "$(dirname "$0")"

# Step 1: Generate HTML from notebook (requires running notebook first)
echo "Step 1: Converting notebook to HTML..."
if command -v jupyter &> /dev/null; then
    cd project
    jupyter nbconvert --to html project.ipynb --output project.html
    cd ..
    echo "✓ HTML file generated"
else
    echo "⚠ Jupyter not found. Please:"
    echo "   1. Open project/project.ipynb in Jupyter"
    echo "   2. Run all cells (should take <1 minute)"
    echo "   3. File → Save and Export Notebook As → HTML"
    echo "   4. Save as project/project.html"
fi

# Step 2: Create zip file
echo ""
echo "Step 2: Creating project.zip..."
cd project
zip -r ../project.zip . -x "*.pyc" "__pycache__/*" "*.ipynb_checkpoints/*"
cd ..
echo "✓ project.zip created"

# Step 3: Check file sizes
echo ""
echo "Step 3: Checking file sizes..."
echo "project.zip size: $(du -h project.zip | cut -f1)"
echo "checkpoint_500k.pt size: $(du -h project/checkpoint_500k.pt | cut -f1)"

echo ""
echo "=========================================="
echo "SUBMISSION READY!"
echo "=========================================="
echo "Files to submit to Canvas:"
echo "  1. FinalReport.pdf (create from your report)"
echo "  2. project.zip (just created)"
echo ""
echo "Make sure project.html exists in project.zip before submitting!"

