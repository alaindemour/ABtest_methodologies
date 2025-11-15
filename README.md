# ABtest_methodologies

A/B testing methodologies and statistical frameworks.

## Notebook Rendering

This repository uses `nbstripout` to keep notebook files clean in version control (outputs and metadata are stripped before committing). To view rendered versions of the notebooks with outputs:

### Automated Rendering

A GitHub Actions workflow automatically converts notebooks to HTML whenever changes are pushed to the main branch:

- **Rendered files location:** `rendered/` directory
- **Trigger:** Automatic on push to main/master when `.ipynb` files change
- **Manual trigger:** Go to Actions tab → "Render Notebooks to HTML" → Run workflow

### Viewing Rendered Notebooks

After the workflow runs, you can view the HTML files:
1. Browse to the `rendered/` directory in the repository
2. Click on any `.html` file
3. Click "Download" or use GitHub's HTML preview

### Local Rendering

To generate HTML versions locally:

```bash
# Install dependencies
pip install nbconvert jupyter

# Render all notebooks
jupyter nbconvert --to html --output-dir rendered *.ipynb

# Render a specific notebook
jupyter nbconvert --to html --output-dir rendered ABmethodologies.ipynb
```

### About nbstripout

The `.gitattributes` file configures Git to automatically strip outputs and metadata from notebooks before committing. This keeps the repository clean and reduces merge conflicts, while the automated workflow ensures rendered versions are always available.