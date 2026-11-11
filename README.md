# Reproduction artifacts

The two reproduction bundles are preserved separately:

- [Commit-then-Sample](CommitThenSample/readme.md)
- [CertiChain](CertiChain/readme.md)

This repository contains the source, configurations, setup programs, tests, and reference summaries. The complete datasets, checkpoints, and detailed experiment outputs are distributed in the matching GitHub Release. Each compressed part is below 2 GiB and has a SHA-256 digest in its payload manifest.

Restore the complete inputs before running either project's setup:

```sh
python3.12 publish_artifacts.py restore
cd CommitThenSample
sh bootstrap.sh
.venv/bin/python reproduce.py smoke
```

Use `--projects CertiChain` to restore only the second bundle. Restoration needs Zstandard (`zstd`), available with `brew install zstd`. Public release downloads do not need a GitHub account. Publication also needs the GitHub CLI (`gh`). Each project's detailed instructions apply after restoration.

The normal Git checkout remains small. Full restoration needs approximately 171 GiB for Commit-then-Sample and 125 GiB for CertiChain, plus one temporary compressed part. The original full local bundles can be used directly without downloading.

## Publish from the source machine

Both destination lines supplied for this task identify `LeoTTTPhat/CommitThenSample`. This publication layout therefore places the two projects in separate directories in that repository, preserving both.

From this directory, run:

```sh
python3.12 publish_artifacts.py publish
```

The command checks SSH access and GitHub CLI write permission before uploading. It pushes the prepared `main` branch, creates or resumes a draft release, streams the payloads into 1800 MiB parts, checks every uploaded asset's remote SHA-256 digest, and removes the temporary local part only after verification. The release becomes public only after both complete payloads have been uploaded. Existing published releases and conflicting assets are never overwritten.

Run this command in a terminal with working GitHub networking and an authenticated `gh` account that has write access to the destination repository. No access token is stored in these files.
