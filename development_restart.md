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