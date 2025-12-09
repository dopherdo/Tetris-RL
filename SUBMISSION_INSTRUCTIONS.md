# Canvas Submission Instructions

## Final Steps to Complete Submission

### 1. Generate project.html from Notebook

You need to run the notebook and export it as HTML:

**Option A: Using Jupyter (Recommended)**
```bash
cd project
jupyter nbconvert --to html project.ipynb --output project.html
```

**Option B: Using Jupyter Notebook/Lab GUI**
1. Open `project/project.ipynb` in Jupyter Notebook or Jupyter Lab
2. Run all cells (should complete in <1 minute)
3. Go to File → Save and Export Notebook As → HTML
4. Save as `project/project.html`

**Option C: Using VS Code/Cursor**
1. Open `project/project.ipynb` in VS Code/Cursor
2. Run all cells
3. Right-click on notebook → "Export As" → HTML
4. Save as `project/project.html`

### 2. Create FinalReport.pdf

Convert your final report (Markdown or Word) to PDF:
- If you have a Markdown file, use Pandoc: `pandoc FinalReport.md -o FinalReport.pdf`
- Or use any PDF converter/Word export

### 3. Create project.zip

Run the helper script:
```bash
./prepare_submission.sh
```

Or manually:
```bash
cd project
zip -r ../project.zip . -x "*.pyc" "__pycache__/*" "*.ipynb_checkpoints/*"
cd ..
```

### 4. Verify Contents

Make sure `project.zip` contains:
- ✅ `project/README.md`
- ✅ `project/project.ipynb`
- ✅ `project/project.html` (with all outputs)
- ✅ `project/checkpoint_500k.pt`
- ✅ `project/requirements.txt`
- ✅ `project/src/` (with all Python files)

### 5. Check File Sizes

- `project.zip` should be reasonable size (checkpoint is ~10-20MB)
- Total data files should be <5MB (the checkpoint is the main file)

### 6. Submit to Canvas

Submit:
1. **FinalReport.pdf** - Your final project report
2. **project.zip** - Contains all code, notebook, and model

## Quick Checklist

- [ ] `project/project.html` exists and has outputs
- [ ] `project/README.md` has one-line descriptions
- [ ] `project/project.ipynb` runs in <1 minute
- [ ] `project.zip` contains all required files
- [ ] `FinalReport.pdf` is ready
- [ ] File sizes are acceptable

## Troubleshooting

**Notebook won't run?**
- Make sure `checkpoint_500k.pt` is in the `project/` directory
- Install dependencies: `pip install -r requirements.txt`
- Check that `src/` directory is present

**HTML has no outputs?**
- You must run all cells in the notebook before exporting
- Make sure to save the notebook after running

**Zip file too large?**
- The checkpoint file is necessary but large
- You can remove `curriculum_wrapper.py` if not needed
- Check for any unnecessary files

