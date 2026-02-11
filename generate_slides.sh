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
with open('presentation.slides.html', 'r') as f:
    html = f.read()

# Inject CSS right before </head>
css_fix = """
<style>
/* Fix: Reveal.js bottom margin cutoff */
.reveal .slides section {
    box-sizing: border-box !important;
    padding-bottom: 40px !important;
    overflow-y: auto !important;
    max-height: 100% !important;
}
.reveal .slide-number {
    bottom: 8px !important;
    right: 8px !important;
}
</style>
"""
html = html.replace('</head>', css_fix + '\n</head>')

# Inject Reveal config override right before </body>
# Must run AFTER Reveal.initialize(), so we place it at the very end
js_fix = """
<script>
// Fix: Override Reveal config after initialization
Reveal.on('ready', function() {
    Reveal.configure({
        margin: 0.12,
        height: 620,
        minScale: 0.2,
        maxScale: 1.5
    });
    // Force re-layout after config change
    Reveal.layout();
});
</script>
"""
html = html.replace('</body>', js_fix + '\n</body>')

with open('presentation.slides.html', 'w') as f:
    f.write(html)

print("Done. Applied CSS and Reveal.configure() fixes.")
PYEOF

echo ""
echo "Generated: presentation.slides.html"
echo "Open in browser to view slides."
