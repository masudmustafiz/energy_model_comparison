# Energy Model Comparison – Calliope, oemof.solph, PyPSA

Master's thesis. Linear LP investment model
office building (8760 h), implemented in three open-source tools with one
shared input dataset.

## Requirements
- Python 3.10+
- calliope, oemof.solph, pypsa, pandas, matplotlib
- LP solver: Gurobi (student academic license) — `pip install gurobipy`             # any solver can be used
## Files
| File | Purpose |
|---|---|
| `input_data_office_medium.csv` | Shared hourly input (demand, PV cf, prices) |
| `Calliope_LP_model.yaml` + `run_calliope.py` | Calliope model |
| `Oemof_LINEAR_LP_NATIVE_SOURCE_BUSES_NO_STORAGE_BOUND.py` | oemof.solph model |
| `PyPSA_LINEAR_LP_NATIVE_SOC_WINDOW_NO...py` | PyPSA model |
| `compare_three_energy_tools_v2.py` | Runs/compares outputs of all three tools |

## Run
pip install calliope oemof.solph pypsa pandas
python run_calliope.py
python Oemof_LINEAR_LP_...py
python PyPSA_LINEAR_LP_...py
python compare_three_energy_tools_v2.py
