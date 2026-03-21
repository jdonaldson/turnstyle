# Turnstyle Project Notes

## Image Editing Workflow

**Commit the raw/clean image before applying transforms** (labels, rounded corners, resizing). This ensures you can always recover the original from git history. Never stack edits on an already-transformed image — always start from the committed clean version.
