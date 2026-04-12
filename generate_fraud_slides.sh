#!/bin/bash
# Generate self-contained Reveal.js slides from fraud_detection_performance.ipynb
#
# Usage: ./generate_fraud_slides.sh
# Output: fraud_detection_performance.slides.html (fully self-contained)
#
# The notebook is executed fresh on each run (nbstripout strips outputs on git ops).

set -e

jupyter nbconvert fraud_detection_performance.ipynb --to slides --execute \
  --TagRemovePreprocessor.remove_input_tags='["remove-input"]'

python3 << 'PYEOF'
import re, base64, mimetypes
from pathlib import Path

with open('fraud_detection_performance.slides.html', 'r') as f:
    html = f.read()

def inline_image(match):
    src = match.group(1)
    if src.startswith('data:'):
        return match.group(0)
    img_path = Path(src)
    if not img_path.exists():
        print(f"  Skipping missing image: {src}")
        return match.group(0)
    mime, _ = mimetypes.guess_type(str(img_path))
    mime = mime or 'image/jpeg'
    b64 = base64.b64encode(img_path.read_bytes()).decode('utf-8')
    print(f"  Inlined: {src} ({img_path.stat().st_size // 1024} KB)")
    return f'src="data:{mime};base64,{b64}"'

html_inlined = re.sub(r'src="([^"]+\.(jpg|jpeg|png|gif|svg|webp))"',
                      inline_image, html, flags=re.IGNORECASE)

with open('fraud_detection_performance.slides.html', 'w') as f:
    f.write(html_inlined)

size_kb = len(html_inlined) // 1024
print(f"Done. Self-contained slide deck: {size_kb} KB")
PYEOF

echo "Generated: fraud_detection_performance.slides.html"
