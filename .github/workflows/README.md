# GitHub Actions Workflows

## Auto Version Bump (`version-bump.yml`)

This workflow automatically bumps the version number in `pyproject.toml` on every push to the `main` branch.

### Features

- **Automatic Versioning**: Increments the patch version (e.g., `0.1.0` → `0.1.1`)
- **Git Tagging**: Creates a git tag for each version (e.g., `v0.1.1`)
- **Infinite Loop Prevention**: Commits include `[skip-version-bump]` to avoid triggering itself
- **Summary Reports**: Generates a workflow summary showing old and new versions
- **Workflow Updates**: Changes to `version-bump.yml` itself won't trigger the workflow (use manual merge to main for workflow updates)

### How It Works

1. **Trigger**: Runs on every push to `main` branch
2. **Version Detection**: Reads current version from `pyproject.toml`
3. **Version Bump**: Increments the patch number (rightmost digit)
4. **Commit**: Creates a commit with the updated version
5. **Tag**: Creates an annotated git tag (e.g., `v0.1.1`)
6. **Push**: Pushes both the commit and tag to the repository

### Skipping Version Bump

To skip the version bump for a specific commit, include `[skip-version-bump]` in your commit message:

```bash
git commit -m "Update documentation [skip-version-bump]"
```

### Version Format

The workflow expects semantic versioning format in `pyproject.toml`:

```toml
[project]
version = "MAJOR.MINOR.PATCH"
```

Example: `0.1.0`, `1.2.3`, `2.0.15`

### Permissions

The workflow requires:
- `contents: write` - To commit changes and push tags

### Dependencies

- Python 3.11
- `toml` package (installed automatically)

### Customization

To bump major or minor versions instead of patch:

1. Edit the Python script in `version-bump.yml`
2. Modify the version bumping logic (lines 60-63)
3. Change `new_patch = int(patch) + 1` to bump different parts

### Troubleshooting

**Issue**: Workflow doesn't trigger
- Ensure the commit doesn't contain `[skip-version-bump]`
- Check that changes are pushed to `main` branch
- Verify workflow file syntax with `yamllint`

**Issue**: Permission denied during push
- Ensure `GITHUB_TOKEN` has write permissions
- Check repository settings → Actions → General → Workflow permissions

**Issue**: Version format not recognized
- Verify `pyproject.toml` uses semantic versioning (`X.Y.Z`)
- Check for extra spaces or invalid characters in version string

**Issue**: Workflow changes don't take effect
- The workflow ignores changes to itself (prevents infinite loops)
- To update the workflow: merge changes to main, then manually trigger or push another change
