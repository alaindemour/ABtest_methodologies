#!/bin/bash
# Generate Reveal.js slides from presentation.ipynb with bottom-margin fix
#
# Usage: ./generate_slides.sh
# Output: presentation.slides.html

set -e

echo "Converting notebook to slides..."
jupyter nbconvert presentation.ipynb --to slides

echo "Applying bottom-margin fix..."
python3 << 'PYEOF'
import re

with open('presentation.slides.html', 'r') as f:
    html = f.read()

# 1. Patch Reveal.initialize() config directly
#    Find the initialization call and inject margin/height overrides
#    Reveal 4.x uses: Reveal.initialize({...}) inside a <script type="module">
#    We need to add our config INTO that initialize call

# Match ONLY Reveal.initialize({ (not mermaid.initialize)
html = re.sub(
    r'(Reveal\.initialize\(\{[^}]*?)height:\s*700',
    r'\1margin: 0.15,\n            height: 600',
    html,
    count=1,
    flags=re.DOTALL
)

# 2. Inject CSS before </head> — pure CSS fixes that don't need JS API
css_fix = """
<style>
/* Fix: Reveal.js bottom content cutoff on macOS Safari/Chrome */
.reveal .slides > section,
.reveal .slides > section > section {
    overflow-y: auto !important;
    height: 100% !important;
}
</style>
"""
html = html.replace('</head>', css_fix + '\n</head>')

with open('presentation.slides.html', 'w') as f:
    f.write(html)

# Verify the patch was applied
if 'height: 600' in html and 'margin: 0.15' in html:
    print("Done. Patched Reveal.initialize(): height 700->600, added margin=0.15.")
else:
    print("WARNING: Could not patch Reveal.initialize().")
    print("Searching for current config...")
    import subprocess
    subprocess.run(['grep', '-n', 'Reveal.initialize', 'presentation.slides.html'])
PYEOF

echo ""
echo "Generated: presentation.slides.html"
echo "Open in browser to view slides."
