# Nicomedia

Nicomedia provides utility functions and profile kernels for astrophysical imaging and catalog-modeling workflows, including King-like and Gaussian-style kernels. Shared numerical implementations are provided by `tdpy` where available.

## Installation

```bash
cd nicomedia
python -m pip install -e .
export NICOMEDIA_PATH=/path/to/nicomedia
```

`NICOMEDIA_PATH` identifies the repository root. Keep runtime inputs in `data/` and generated pipeline outputs in `visuals/`; both directories are ignored by Git.

## Minimal workflow

A runnable compatibility example evaluates the same analytic double-King profile through Nicomedia and TDpy:

```bash
python examples/double_king_compatibility.py --typefileplot png
```

![Nicomedia and TDpy double-King profile compatibility](examples/double_king_compatibility.png)

The upper panel shows the shared analytic profile for explicit kernel parameters. The lower panel shows the absolute numerical difference. It is exactly zero, demonstrating that Nicomedia preserves the TDpy implementation rather than maintaining a redundant kernel.

## Dependencies

The package sits on the standard scientific stack and relies primarily on the shared ecosystem utilities provided by `tdpy`.
