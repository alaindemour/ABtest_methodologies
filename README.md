# ABtest_methodologies

A/B testing methodologies and statistical frameworks.

## 📖 View Rendered Notebooks

**[View all rendered notebooks on GitHub Pages →](https://alaindemour.github.io/ABtest_methodologies/)**


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

```bash
jupyter nbconvert presentation.ipynb --to slides
```
