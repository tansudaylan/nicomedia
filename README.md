# Nicomedia

Nicomedia evaluates King-like and Gaussian point-spread profiles for astrophysical imaging and catalog models at scalar or array-valued coordinates.

## Installation

```bash
cd nicomedia
python -m pip install -e .
export NICOMEDIA_PATH=/path/to/nicomedia
```

`NICOMEDIA_PATH` identifies the repository root. Keep runtime inputs in `data/` and generated pipeline outputs in `visuals/`; both directories are ignored by Git.

## Double-King profile

The example evaluates a double-King point-spread profile with explicit core and wing parameters:

```bash
python examples/double_king_compatibility.py --typefileplot png
```

![Nicomedia double-King point-spread profile](examples/double_king_compatibility.png)

The upper panel shows the radial decline of the analytic profile. The lower panel confirms numerical agreement between the Nicomedia and TDpy evaluations across the plotted radii.

## Dependencies

Nicomedia uses NumPy for array calculations and TDpy for numerical and plotting operations.
