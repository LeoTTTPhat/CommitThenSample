Complete input datasets, model snapshots, prepared data, and recorded experiment checkpoints for both reproduction projects.

The assets contain only the reproduction bundles. Compressed streams are split into numbered parts smaller than 2 GiB. The project-specific payload JSON files record part order, byte sizes, and SHA-256 digests. Use `python3.12 publish_artifacts.py restore` from the matching source checkout to download, verify, and restore the complete bundles.
