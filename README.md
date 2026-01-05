
# Number Entanglement

This repository computes distributions of number entanglement for bipartite
systems and saves diagnostic plots.

**Requirements**
- Python 3.8 or newer
- See `requirements.txt` for required packages

**Setup (Windows PowerShell)**
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

**Run**
From the project root (where `number_entanglement.py` lives) run:

```powershell
python number_entanglement.py
```

This will generate PNG files in the current directory. Filenames include a
suffix determined by the script options (e.g. `_qubits`, `_add`). The histogram
files are named like `hist_chi_D_<dim><suffix>` and summary plots are
`chi_k<suffix>`, `chi_std<suffix>`.

The code contained in `unit_test.py` is used to insure the code behaves correctly, so it
should not be needed by the end user.

**What to change between executions**

- Random seed / reproducibility:
	- Edit the RNG definition at [number_entanglement.py](number_entanglement.py#L30)
		(line 30). By default it uses an OS-seeded generator.

- Main run parameters (change these to alter what the script computes):
	- `N_STATES` (number of random separable states to generate)
	- `qubits` (True to use qubit-number operators, False for multilevel)
	- `add` (True to use tensor-sum operator variant, False for tensor product operator)
	- `dims_a` (array of subsystem dimensions to de computed)

	The parameters are defined together in the `if __name__ == "__main__"` block
	at [number_entanglement.py](number_entanglement.py#L406-L410). Example lines:

```python
N_STATES = 10000
qubits = True
add = False
dims_a = np.array(list(range(1, 4)))
dims_a = 2**dims_a
```

Change `N_STATES` to increase/decrease sample size (larger -> slower, smoother
statistics). Modify `dims_a` to change which subsystem sizes are tested.

**Outputs**
- Histogram per dimension: `hist_chi_D_<dim><suffix>` (PNG)
- Fitted parameter plots: `chi_k<suffix>` and `chi_std<suffix>` (PNG)
