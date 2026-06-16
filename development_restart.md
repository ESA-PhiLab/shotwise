Initialization:
- Prepare virtual env:
cd ~/dev/shotwise
source .venv/bin/activate

- Tests and precommit
pytest -q
pre-commit run --all-files

- Commit:
git add .
git commit -m "message"

- Branches handling
git checkout main
git merge --no-ff old_branch
pytest -q
ruff check .
pre-commit run --all-files
git branch -d old_branch
