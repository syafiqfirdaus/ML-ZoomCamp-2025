# Git LFS Guide

Git LFS (Large File Storage) is an extension for Git that allows you to version large files efficiently without bloating your repository.

## Installation

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# Download from https://git-lfs.github.com/
```

## Setup for Your Repository

### 1. Initialize Git LFS (one-time setup)

```bash
git lfs install
```

### 2. Track Large Files

For your case with `data.zip`:

```bash
# Track all .zip files
git lfs track "*.zip"

# Or track specific file
git lfs track "Homework/08-Deep Learning/data.zip"
```

This creates/updates a `.gitattributes` file.

### 3. Add the .gitattributes file

```bash
git add .gitattributes
```

### 4. Add your large files

```bash
git add "Homework/08-Deep Learning/data.zip"
git commit -m "Add data.zip with Git LFS"
git push
```

## Migrating Existing Large Files

If you've already committed large files (like we just did), you can migrate them to LFS:

```bash
# Install git-lfs if not already done
git lfs install

# Track the file type
git lfs track "*.zip"

# Migrate existing files in history
git lfs migrate import --include="*.zip" --everything

# Force push (WARNING: rewrites history)
git push --force
```

## Common Commands

```bash
# See which files are tracked by LFS
git lfs ls-files

# Check LFS status
git lfs status

# Pull LFS files
git lfs pull

# See what's being tracked
cat .gitattributes
```

## Benefits

- ✅ Keeps repository size small
- ✅ Faster cloning and fetching
- ✅ Better performance with large binary files
- ✅ GitHub supports LFS (free tier: 1GB storage, 1GB bandwidth/month)

## Note for Your Current Situation

Since `data.zip` is already pushed, you have two options:

1. **Leave it as is** - It's already uploaded and working fine. GitHub just warns about the size.
2. **Migrate to LFS** - Use the migration commands above to move it to LFS (requires force push).

For future homeworks with large files, set up LFS tracking before the first commit!
