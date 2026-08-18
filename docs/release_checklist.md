# Public Release Checklist

Complete this checklist before the first GitHub push and before each tagged release.

## Repository metadata

- Run `python scripts/set_repository_owner.py YOUR_GITHUB_ID` to replace `<ORG_OR_USER>` in `README.md` and `CITATION.cff`.
- Confirm the author names, release date, version, and software license.
- Create an empty GitHub repository without an auto-generated README, license, or `.gitignore`.

## Protected-data check

- Do not add DICOM, NIfTI, masks, model checkpoints, reports, review spreadsheets, patient identifiers, institutional paths, or logs.
- Confirm that examples remain synthetic and contain the `SYNTHETIC` marker.
- Keep the repository private until institutional code-release and model-weight decisions are complete.

## Local verification

```bash
python scripts/check_public_release.py
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
git status --short
```

## First commit and push

Configure your own Git identity if it is not already configured:

```bash
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
git commit -m "Initial public research-code release"
git remote add origin https://github.com/<ORG_OR_USER>/stroke-dwi-report-generation.git
git push -u origin main
```

Do not paste credentials into source files or chat logs. Authenticate using GitHub Credential Manager, SSH, or the browser prompt.
