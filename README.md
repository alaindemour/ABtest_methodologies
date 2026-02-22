# ABtest_methodologies

A/B testing methodologies and statistical frameworks.

## 📖 View Rendered Notebooks

**[View all rendered notebooks on GitHub Pages →](https://alaindemour.github.io/ABtest_methodologies/)**


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

To produce the slide deck it relies on

```bash
jupyter nbconvert presentation.ipynb --to slides
jupyter nbconvert presentation.ipynb --to slides --TagRemovePreprocessor.remove_input_tags='["remove-input"]'
```

but  this is better to actually have all the bells and whistles

```
./generate_slides.sh
```


