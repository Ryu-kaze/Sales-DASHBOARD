#!/bin/bash
# ── Force update app.py on GitHub ──────────────────────────

GITHUB_USERNAME="Ryu-kaze"
REPO_NAME="saleslens-forecast"
TOKEN="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"

echo "🔄 Pushing updated app.py to GitHub..."

cd "$(dirname "$0")"

git add app.py requirements.txt
git commit -m "fix: use st.secrets.get() with fallback key — fixes KeyError on Streamlit Cloud"
git push "https://$TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git" main

echo ""
echo "✅ Done! Go to your Streamlit Cloud app and click 'Reboot app'."
echo "   https://share.streamlit.io"
