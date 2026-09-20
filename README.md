# q-commutator computations

Code for the numerical and exact computations used in the paper.

Paper: coming soon

## Files

- `numerical_experiment.py` - runs the alternating optimization for the traceless-traceless numerical experiment.
- `verify_exact_examples.py` - checks the exact 3x3 and 5x5 counterexamples.
- `results.csv` - numerical results for the tested `(n,q)` grid.
- `requirements.txt` - Python dependencies.

## Run

```bash
pip install -r requirements.txt
python numerical_experiment.py
python verify_exact_examples.py
```

Python: 3.9.13

Random seed: 20260919
