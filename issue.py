import os
from typing import List, Dict

from github import Github


def load_env() -> None:
    """Load .env if python-dotenv is available; ignore if not installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def ensure_labels(repo) -> None:
    """Create standard labels if they don't exist (idempotent)."""
    label_colors = {
        "good first issue": "F1E05A",
        "enhancement": "84b6eb",
        "documentation": "006b75",
        "bug": "d73a4a",
        "infra": "5319E7",
    }
    existing = {l.name for l in repo.get_labels()}
    for name, color in label_colors.items():
        if name not in existing:
            repo.create_label(name=name, color=color)


def curated_issues() -> List[Dict]:
    """Return a list of issues to create with title/body/labels."""
    return [
        dict(
            title="Auto-load .env and document Azure/OpenAI config",
            body=(
                "Problem\n"
                "The app requires env vars (USE_GPT_DIAGRAM, AZURE_OPENAI_*), but users must manually source .env.\n\n"
                "Solution\n- Load .env via python-dotenv in app/streamlit_app.py (and api/main.py).\n"
                "- Update README + .env.example with Azure fields and comments.\n\n"
                "Acceptance\n- .env auto-loads; missing keys -> graceful fallback; docs updated.\n"
            ),
            labels=["enhancement", "documentation", "good first issue"],
        ),
        dict(
            title="Show UI warning if USE_GPT_DIAGRAM=true but Azure env incomplete",
            body=(
                "Add a visible Streamlit warning when USE_GPT_DIAGRAM=true but any of:\n"
                "- AZURE_OPENAI_ENDPOINT\n- AZURE_OPENAI_API_KEY (or AZURE_OPEN_AI_KEY)\n- AZURE_OPENAI_DEPLOYMENT\n- AZURE_OPENAI_API_VERSION\n"
                "is missing. Still fall back to local detection.\n"
            ),
            labels=["enhancement", "good first issue"],
        ),
        dict(
            title="Audit Streamlit APIs for compatibility (use_column_width vs use_container_width)",
            body=(
                "Replace deprecated/unsupported args (e.g., st.image(use_container_width=...)) with supported ones\n"
                "(use_column_width=True) across the app.\n"
            ),
            labels=["bug", "good first issue"],
        ),
        dict(
            title="Improve error messages and logging for GPT diagram detection",
            body=(
                "- When GPT call fails or returns no boxes, log reason (rate limit, auth, timeout).\n"
                "- Surface a concise UI note and fall back to local detection.\n"
            ),
            labels=["enhancement"],
        ),
        dict(
            title="Add optional overlay to visualize OCR lines and diagram boxes",
            body=(
                "Add a 'Debug overlays' toggle to display OCR text line boxes and diagram bounding boxes (GPT/local).\n"
            ),
            labels=["enhancement"],
        ),
        dict(
            title="Document PyTorch install on macOS (CPU wheels) in README",
            body=(
                "Add a section showing:\n"
                "pip install --upgrade pip setuptools wheel\n"
                "pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cpu\n"
                "Explain why versions are pinned in requirements.txt.\n"
            ),
            labels=["documentation", "good first issue"],
        ),
        dict(
            title="Add pre-commit with black, isort, flake8 and run in CI",
            body=(
                "- Add .pre-commit-config.yaml with black, isort, flake8.\n"
                "- Update README to install pre-commit.\n"
                "- Add CI job that runs pre-commit on PRs.\n"
            ),
            labels=["infra", "enhancement", "good first issue"],
        ),
        dict(
            title="Add basic tests for OCR pipeline and FastAPI endpoint",
            body=(
                "- Unit test for src/mcq_extractor.extract_mcq() with a tiny sample.\n"
                "- API test in tests/test_api.py to ensure /extract works and returns expected schema.\n"
                "- Add small test assets.\n"
            ),
            labels=["enhancement"],
        ),
        dict(
            title="GitHub Actions: run tests and lint on PR",
            body=(
                "- Setup Python matrix (3.9+) with cache.\n"
                "- Install -r requirements.txt.\n"
                "- Run pytest and pre-commit.\n"
            ),
            labels=["infra", "enhancement"],
        ),
        dict(
            title="Streamlit: show diagram crops in a responsive grid with download",
            body=(
                "- Display diagram crops in columns.\n"
                "- Add a 'Download all crops (.zip)' button.\n"
            ),
            labels=["enhancement"],
        ),
        dict(
            title="Robust diagram crop saving and metadata JSON",
            body=(
                "- Save crops to data/diagrams with stable filenames.\n"
                "- Write a sidecar JSON per crop with bbox, source image hash, detector='gpt|local'.\n"
            ),
            labels=["enhancement"],
        ),
        dict(
            title="Config guardrails: validate env at startup and print summary",
            body=(
                "- On app start, print detected config: device, GPT enabled?, endpoint set?, etc.\n"
                "- Avoid crashing; fall back to local detection when unset.\n"
            ),
            labels=["enhancement", "good first issue"],
        ),
        dict(
            title="README: Quickstart and sample images section",
            body=(
                "- Add 'Quickstart' with one copy-paste block to run the app.\n"
                "- Include 1-2 sample images or link to public images for testing.\n"
            ),
            labels=["documentation", "good first issue"],
        ),
        dict(
            title="Docker: slim base, multi-stage build, add healthcheck",
            body=(
                "- Multi-stage to slim runtime image.\n"
                "- Add HEALTHCHECK and document how to run Streamlit in container.\n"
            ),
            labels=["infra", "enhancement"],
        ),
        dict(
            title="Add CLI to batch-extract MCQs from a folder",
            body=(
                "- Provide a CLI to process a directory of images and save JSON + crops.\n"
            ),
            labels=["enhancement"],
        ),
    ]


def main() -> None:
    load_env()
    token = os.getenv("GITHUB_TOKEN")
    repo_slug = os.getenv("REPO_SLUG")
    if not token or not repo_slug:
        raise SystemExit("Missing GITHUB_TOKEN or REPO_SLUG. Export them or add to .env.")

    gh = Github(token)
    repo = gh.get_repo(repo_slug)

    ensure_labels(repo)

    # Collect existing issue titles (all states) to avoid duplicates
    existing_titles = set()
    for issue in repo.get_issues(state="all"):
        existing_titles.add(issue.title.strip())

    created = 0
    for spec in curated_issues():
        title = spec["title"].strip()
        if title in existing_titles:
            continue
        labels = [repo.get_label(l) for l in spec["labels"]]
        repo.create_issue(title=title, body=spec.get("body", ""), labels=labels)
        created += 1

    print(f"Created {created} new issue(s) in {repo_slug}")


if __name__ == "__main__":
    main()
 