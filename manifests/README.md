# Artifact manifest

`final_artifacts.sha256` records the frozen datasets, canonical graph caches,
50k checkpoints, and evidence packages used by the validation study.

The listed paths are logical release paths. Large files are intentionally not
committed to Git and should be distributed through a versioned release or a
research-data archive. After downloading an artifact, verify it with:

```bash
shasum -a 256 /path/to/artifact
```

The validated branch-token checkpoint is the
`checkpoints/ebt/full/latest.pt` member of
`branch_token_50k_reproducibility.zip`. An earlier local file named
`etb_50K_latest.pt` has a different hash and is not the checkpoint recorded
here.

At this project stage, evaluation results are validation-only and the test
split has not been accessed.

