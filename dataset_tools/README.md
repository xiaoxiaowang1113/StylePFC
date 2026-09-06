# StyleFamilyBench construction tools

These scripts reconstruct the released 240-image StyleFamilyBench selection or
create a new candidate selection under the same protocol. The complete commands
and directory layout are documented in [`../DATASET.md`](../DATASET.md).

## Scripts

| Script | Purpose |
| --- | --- |
| `download_wikiart.py` | Export the 81,444-image, 27-label WikiArt source into style directories. |
| `select_wikiart_styles.py` | Reconstruct the released 50 candidates per family, or create a new stratified sample. |
| `review_candidates.py` | Generate reason-aware HTML review pages or apply the released rejection manifests. |
| `rename_to_digits.py` | Build the final six-by-40 PNG benchmark and write mapping/checksum metadata. |
| `verify_stylefamilybench.py` | Validate counts, names, dimensions, mappings, and generated checksums without modifying files. |

The `manifests/` directory contains the released candidate list, filter
statistics, six manual rejection lists, review log, and final filename mapping.
These small metadata files are required to reproduce the same selection; source
images are not redistributed.

## Exact reconstruction versus resampling

By default, `select_wikiart_styles.py` uses
`manifests/selection_log.json`. This is the exact reconstruction path. Pass
`--resample` only when intentionally creating a different candidate set. A
resampled set is not the released StyleFamilyBench benchmark and must be reported
as a separate dataset version.

All scripts refuse to merge their results into a non-empty output directory.
Choose a new output path for every rebuild so stale files cannot silently alter
the benchmark.
