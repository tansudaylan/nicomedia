# nicomedia
A library of numerical routines in astrophysics.

## Current status

This repository is best treated as a compatibility wrapper around the canonical shared numerical layer in `tdpy`. It contains numerical utilities and PSF kernels that overlap with the lower-level implementations in `tdpy`, but it should not become a second source of truth for new implementations.

For new scientific code, prefer adding or extending the functionality in `tdpy` and then reusing that implementation from the ecosystem-level library.

## Scientific scope

The routines in this package correspond to numerical and PSF-related astrophysics utilities, especially profile calculations such as Gaussian and King-type kernels used in image and catalog modeling workflows.

## Relationship to the wider ecosystem

- Canonical utility layer: `tdpy`
- Compatibility wrapper: `nicomedia`
- Decision: keep only if it provides a clear wrapper or workflow-specific convenience layer; otherwise prefer merging or retiring redundant code in favor of `tdpy`.
