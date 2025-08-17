#!/usr/bin/env python3
import os
from github import Github

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

token = os.getenv("GITHUB_TOKEN")
repo_slug = os.getenv("REPO_SLUG")
if not token or not repo_slug:
    raise SystemExit("Missing GITHUB_TOKEN or REPO_SLUG. Export them or add to .env.")

g = Github(token)
repo = g.get_repo(repo_slug)

# Enable Discussions and Wiki
repo.edit(has_discussions=True, has_wiki=True)
print(f"Enabled Discussions and Wiki for {repo_slug}")
