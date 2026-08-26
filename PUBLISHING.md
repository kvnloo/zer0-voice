# Publishing zer0-voice

1. Push a temporary `extraction-review` branch, never a default branch.
2. Compare `git log --follow`, source digests, and this repo's tests
   against the `zer0-integration-before-nanoservice-split` tag in the integration repository.
3. Add ownership rules, a changelog, and a release tag.
4. Package and pin `zer0-harness-spine` first; the other repositories
   depend on an exact released version.
5. Make `python -m unittest guards.test_service_boundaries` mandatory.
6. Delete extracted paths from the integration repository only after
   every review branch reproduces the end-to-end canary.

## Root-hoisted mirror provenance

The public mirror hoists `voice/` files to its repository root. Runtime adapter
files are hoisted alongside them from `zer0-harness-spine` commit
`30c5d3fc312e47a431bec1e96f5c3848456737c9`, the integration parent of the
continuous-publish extraction. `repository_layout.py` is the single mapping
between these mirror paths and the canonical paths retained in release bundle
manifests. Run `python3 ci/verify_clean_clone.py` after committing to prove the
tracked clean-clone fixture.
