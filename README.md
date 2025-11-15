# ABtest_methodologies

A/B testing methodologies and statistical frameworks.

## 📖 View Rendered Notebooks

**[View all rendered notebooks on GitHub Pages →](https://alaindemour.github.io/ABtest_methodologies/)**

This repository uses `nbstripout` to keep notebook source files clean in version control (outputs and metadata are stripped before committing). Fully rendered versions with all outputs are automatically published to GitHub Pages.

### How It Works

1. **Local Development:** Edit notebooks in Jupyter with full outputs visible
2. **Commit:** Push to main - `nbstripout` automatically strips outputs before committing
3. **Auto-Render:** GitHub Actions converts notebooks to HTML and publishes to GitHub Pages
4. **View:** Anyone can view the rendered notebooks at the GitHub Pages URL

### Automated Rendering

A GitHub Actions workflow automatically converts notebooks to HTML whenever changes are pushed to main:

- **Published to:** GitHub Pages (`gh-pages` branch)
- **URL:** `https://alaindemour.github.io/ABtest_methodologies/`
- **Trigger:** Automatic on push to main/master when `.ipynb` files change
- **Manual trigger:** Go to Actions tab → "Render Notebooks to HTML" → Run workflow

### Direct Notebook Links

Once published, notebooks can be accessed at:
- `https://alaindemour.github.io/ABtest_methodologies/ABmethodologies.html`
- Add more notebook links as you create them

### Local Rendering (Optional)

To generate HTML versions locally for testing:

```bash
# Install dependencies
pip install nbconvert jupyter

# Render all notebooks
jupyter nbconvert --to html --output-dir rendered *.ipynb

# Render a specific notebook
jupyter nbconvert --to html --output-dir rendered ABmethodologies.ipynb
```

### About nbstripout

The `.gitattributes` file configures Git to automatically strip outputs and metadata from notebooks before committing. This keeps the repository clean and reduces merge conflicts, while the automated workflow ensures rendered versions are always available on GitHub Pages.