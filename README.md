# CommitThenSample reproduction artifact

This checkout publishes CommitThenSample independently to `LeoTTTPhat/CommitThenSample`. It contains the source, configurations, tests, setup programs, and reference summaries. Complete datasets, model snapshots, and experiment checkpoints are distributed as verified parts in the matching GitHub Release.

## Publish

From this directory on the source machine:

```sh
../.venv/bin/python publish_artifacts.py publish
```

This pushes the prepared `main` branch, creates or resumes a draft release, streams the CommitThenSample payload into parts below 2 GiB, verifies every uploaded asset's SHA-256 digest, and publishes the complete release. It needs working SSH access, Zstandard (`zstd`), and a GitHub CLI (`gh`) account with write access. The destination repository must already exist. No access token is saved in this checkout.

## Restore and reproduce

After cloning the repository:

```sh
python3.12 publish_artifacts.py restore
cd CommitThenSample
sh bootstrap.sh
.venv/bin/python reproduce.py smoke
```

Restoration needs Zstandard and approximately 171 GiB of disk space, plus one temporary compressed part. Public downloads do not need a GitHub account. Follow [the reproduction instructions](CommitThenSample/readme.md) after restoration.
