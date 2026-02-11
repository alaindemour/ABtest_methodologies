#!/bin/bash
# Generate Reveal.js slides from presentation.ipynb
#
# Usage: ./generate_slides.sh
# Output: presentation.slides.html

set -e
jupyter nbconvert presentation.ipynb --to slides
echo "Generated: presentation.slides.html"
