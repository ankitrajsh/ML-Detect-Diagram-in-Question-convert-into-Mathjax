#!/usr/bin/env bash
set -euo pipefail

: "${REPO_SLUG:?Set REPO_SLUG, e.g. export REPO_SLUG=owner/repo}"

WIKI_DIR=repo.wiki

# Clone wiki repo
rm -rf "$WIKI_DIR"

# Prefer token-authenticated HTTPS if available
GIT_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -n "${GIT_TOKEN}" ]]; then
  AUTH_URL="https://x-access-token:${GIT_TOKEN}@github.com/${REPO_SLUG%.git}.wiki.git"
else
  # Fallback to SSH if available
  if ssh -T git@github.com -o StrictHostKeyChecking=accept-new 2>/dev/null; then
    AUTH_URL="git@github.com:${REPO_SLUG%.git}.wiki.git"
  else
    AUTH_URL="https://github.com/${REPO_SLUG%.git}.wiki.git"
    echo "Warning: No GH_TOKEN/GITHUB_TOKEN and SSH not configured. Using public HTTPS may fail if repo is private or pushing requires auth."
  fi
fi

echo "Cloning $AUTH_URL"
if ! git clone "$AUTH_URL" "$WIKI_DIR"; then
  echo "If the wiki repo doesn't exist yet, it will be created on first push. Initializing a new wiki repo locally."
  mkdir -p "$WIKI_DIR"
  pushd "$WIKI_DIR" >/dev/null
  git init
  git remote add origin "$AUTH_URL" || true
else
  pushd "$WIKI_DIR" >/dev/null
fi

# Create pages
cat > Home.md << 'EOF'
# ML-Detect-Diagram-in-Question-convert-into-MathJax Wiki

Welcome! Start here to learn how to run the app, configure Azure GPT diagram detection, and contribute.
- [[Getting-Started]]
- [[Configuration]]
- [[Contributing]]
- [[FAQ]]
EOF

cat > Getting-Started.md << 'EOF'
# Getting Started
1. Clone the repo and create a venv.
2. pip install -r requirements.txt
3. streamlit run app/streamlit_app.py
EOF

cat > Configuration.md << 'EOF'
# Configuration
- USE_GPT_DIAGRAM=true
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY (or AZURE_OPEN_AI_KEY)
- AZURE_OPENAI_DEPLOYMENT
- AZURE_OPENAI_API_VERSION
EOF

cat > Contributing.md << 'EOF'
# Contributing
- Check open issues and discussions.
- Open a PR with a clear description.
- Use pre-commit if configured and ensure tests pass.
EOF

cat > FAQ.md << 'EOF'
# FAQ
Q: How to install PyTorch on macOS CPU?
A: pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cpu
EOF

# Push
git add .
git commit -m "Initialize wiki: Home, Getting Started, Configuration, Contributing, FAQ" || true
if ! git push -u origin HEAD:main 2>/dev/null; then
  # Some wikis use 'master'
  git branch -M master || true
  git push -u origin HEAD:master || git push -u origin master
fi

popd >/dev/null
echo "Wiki initialized and pushed."
