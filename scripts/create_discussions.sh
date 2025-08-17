#!/usr/bin/env bash
set -euo pipefail

: "${REPO_SLUG:?Set REPO_SLUG, e.g. export REPO_SLUG=owner/repo}"

# Requires: gh auth login

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is not installed."
  echo "Install on macOS: brew install gh"
  echo "Then run: gh auth login"
  exit 127
fi

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "Error: gh is not authenticated. Run: gh auth login"
  exit 1
fi

echo "Creating Discussions on $REPO_SLUG"

gh discussion create -R "$REPO_SLUG" \
  --title "Project Roadmap" \
  --category "General" \
  --body "High-level goals, milestones, and priorities. Please comment with suggestions."

gh discussion create -R "$REPO_SLUG" \
  --title "Q&A / Help" \
  --category "General" \
  --body "Ask questions or request help here. Maintainers and community members will assist."

gh discussion create -R "$REPO_SLUG" \
  --title "Ideas and Feature Requests" \
  --category "General" \
  --body "Propose and discuss feature ideas. Upvote helpful suggestions."

echo "Done."
