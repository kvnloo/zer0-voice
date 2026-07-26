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
