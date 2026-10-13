"""
================================================================================
 run_calliope.py — Building Energy-System Investment Optimisation (pure LP)

 PV + Battery + CHP (gas) + Heat Pump + AUX Gas Boiler + TES + Grid
 under a fixed UPFRONT / OVERNIGHT CAPEX BUDGET.

 Tool    : Calliope 0.6.10 (Pyomo backend)
 Solver  : Gurobi
 Input   : input_data_office_medium.csv + Calliope_LP_model.yaml
 Output  : Calliope_LP_results.xlsx

 Modelling rule used in this thesis version
 ------------------------------------------
 * Python is the SINGLE SOURCE for numerical model parameters.
 * YAML keeps the model structure/topology and native Calliope 0.6.10 fields.
 * Numerical placeholders in YAML are overwritten here with override_dict.
 * Native Calliope constraints are used wherever possible.
 * The only custom Pyomo constraint is the raw overnight-CAPEX budget because
   Calliope's objective annualises investment costs using lifetime + WACC.
================================================================================
"""

from __future__ import annotations

import pathlib
import pandas as pd
import calliope
from pyomo.environ import Constraint, SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition


# ==============================================================================
# SECTION 0 — VERSION GUARD
# ==============================================================================

REQUIRED_CALLIOPE_VERSION = "0.6.10"
installed_calliope_version = getattr(calliope, "__version__", "unknown")
if installed_calliope_version != REQUIRED_CALLIOPE_VERSION:
    raise RuntimeError(
        f"This thesis model is written for Calliope {REQUIRED_CALLIOPE_VERSION}, "
        f"but the installed version is {installed_calliope_version}. "
        "Do not run it with Calliope 0.7+ syntax/behaviour."
    )


# ==============================================================================
# SECTION 1 — FILES AND INPUT DATA
# ==============================================================================

DATA_DIR   = pathlib.Path(__file__).resolve().parent
MODEL_YAML = DATA_DIR / "Calliope_LP_model.yaml"
INPUT_CSV  = DATA_DIR / "input_data_office_medium.csv"
OUT_XLSX   = DATA_DIR / "Calliope_LP_results.xlsx"

if not MODEL_YAML.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_YAML}")
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

data = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)

required_columns = {
    "heat_demand_kW",
    "el_demand_kW",
    "pv_cf",
    "grid_price_eur_kWh",
    "feedin_tariff_eur_kWh",
}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

MAX_HEAT_KW = float(data["heat_demand_kW"].max())
MAX_EL_KW   = float(data["el_demand_kW"].max())


# ==============================================================================
# SECTION 2 — ALL NUMERICAL MODEL PARAMETERS
# ==============================================================================

# budget 
WACC   = 0.04
BUDGET = 300000.0  # EUR — raw upfront / overnight CAPEX ceiling

# PV
CAPEX_PV      = 1100.0       # EUR/kWp
LIFE_PV       = 25.0         # years
PV_EFF        = 1.0
NET_LIVING_M2 = 1600.0       # office floor area
PV_AREA_RATIO = 0.40
M2_PER_KWP    = 6.0
PV_MAX        = NET_LIVING_M2 * PV_AREA_RATIO / M2_PER_KWP

# Battery
CAPEX_BAT   = 450.0          # EUR/kWh
LIFE_BAT    = 15.0           # years
BAT_EFF     = 0.95           # one-way; round-trip = 0.95^2 = 0.9025
BAT_LOSS    = 0.0            # fraction/hour
BAT_C_RATE  = 1.0            # kW per kWh (native max ratio)
BAT_MIN_SOC = 0.10           # native minimum SOC fraction
BAT_MAX_SOC = 1.00           # native Calliope upper bound: storage <= storage_cap
BAT_OM_PROD = 0.001          # EUR/kWh discharged; small anti-cycling cost

# Thermal Energy Storage (TES)
CAPEX_TES   = 50.0           # EUR/kWh_th
LIFE_TES    = 20.0           # years
TES_EFF     = 0.98           # one-way
TES_LOSS    = 0.001          # fraction/hour
TES_C_RATE  = 0.25           # kW_th per kWh_th (4-hour maximum rate)
TES_OM_PROD = 0.001          # EUR/kWh_th discharged; anti-cycling cost

# Heat Pump
CAPEX_HP = 1200.0            # EUR/kW_th
LIFE_HP  = 20.0              # years
COP_HP   = 3.0               # kWh_th out / kWh_el in
HP_MAX   = MAX_HEAT_KW       # kW_th
HP_RAMP  = 0.50              # fraction of installed energy_cap per hour

# CHP gas engine
CAPEX_CHP     = 1500.0       # EUR/kW_el (primary output capacity)
LIFE_CHP      = 20.0         # years
ETA_EL_CHP    = 0.35
ETA_TH_CHP    = 0.45
CHP_HEAT_RATIO = ETA_TH_CHP / ETA_EL_CHP  # heat_out / el_out
CHP_MAX_EL    = (MAX_HEAT_KW / ETA_TH_CHP) * ETA_EL_CHP
CHP_RAMP      = 0.25         # fraction of installed primary energy_cap per hour

# Auxiliary gas boiler
CAPEX_AUX = 150.0            # EUR/kW_th
LIFE_AUX  = 20.0             # years
AUX_EFF   = 0.90
AUX_MAX   = MAX_HEAT_KW      # kW_th

# Fuels / operating penalties
GAS_PRICE = 0.07             # EUR/kWh_gas; charged once at gas_source

# Make AUX an emergency/backup heat source economically, without charging gas
# twice. The gas_source already charges GAS_PRICE per kWh_gas. Therefore this
# is an ADDITIONAL cost per kWh_th of AUX heat output only.
HP_WORST_CASE_HEAT_COST = float(data["grid_price_eur_kWh"].max()) / COP_HP
AUX_BASE_HEAT_COST      = GAS_PRICE / AUX_EFF
AUX_BACKUP_MARGIN       = 0.01  # EUR/kWh_th above worst-case grid-HP heat
AUX_BACKUP_PENALTY = max(
    0.0,
    HP_WORST_CASE_HEAT_COST - AUX_BASE_HEAT_COST,
) + AUX_BACKUP_MARGIN

# Structural numerical constants intentionally controlled from Python
WIRE_EFF                = 1.0
OPTIONAL_SINK_LIMIT     = 1.0e6   # kWh/h absolute upper magnitude
HEAT_DUMP_PENALTY       = 0.0001  # EUR/kWh_th dumped


# ==============================================================================
# SECTION 3 — GENERATE CALLIOPE TIMESERIES FILES
# ==============================================================================
# Column name = location name ('site') — must match locations: in YAML.
# Demand CSVs are NEGATIVE because Calliope demand resources consume carriers.
# ==============================================================================

loc = "site"

pd.DataFrame({"timestamp": data.index, loc: data["pv_cf"].values}) \
    .set_index("timestamp").to_csv(DATA_DIR / "pv_cf.csv")

pd.DataFrame({"timestamp": data.index, loc: data["grid_price_eur_kWh"].values}) \
    .set_index("timestamp").to_csv(DATA_DIR / "grid_price.csv")

pd.DataFrame({
    "timestamp": data.index,
    loc: -data["feedin_tariff_eur_kWh"].values,
}).set_index("timestamp").to_csv(DATA_DIR / "neg_feedin.csv")

pd.DataFrame({"timestamp": data.index, loc: -data["el_demand_kW"].values}) \
    .set_index("timestamp").to_csv(DATA_DIR / "el_demand.csv")

pd.DataFrame({"timestamp": data.index, loc: -data["heat_demand_kW"].values}) \
    .set_index("timestamp").to_csv(DATA_DIR / "heat_demand.csv")


# ==============================================================================
# SECTION 4 — BUILD MODEL WITH OVERRIDE_DICT
# ==============================================================================
# The override paths intentionally point at model-wide `techs.*` definitions.
# With one location this is clearer than repeating the same numerical values in
# `locations.site.techs.*`. YAML therefore remains a structural template.
# ==============================================================================

T = "techs."

override_dict = {
    # --------------------------------------------------------------------------
    # PV
    # --------------------------------------------------------------------------
    T + "pv.constraints.energy_cap_max": PV_MAX,
    T + "pv.constraints.energy_eff": PV_EFF,
    T + "pv.constraints.lifetime": LIFE_PV,
    T + "pv.costs.monetary.interest_rate": WACC,
    T + "pv.costs.monetary.energy_cap": CAPEX_PV,

    # --------------------------------------------------------------------------
    # Gas source
    # --------------------------------------------------------------------------
    T + "gas_source.costs.monetary.om_prod": GAS_PRICE,

    # --------------------------------------------------------------------------
    # Optional export / dump sinks
    # --------------------------------------------------------------------------
    T + "grid_export.constraints.resource": -OPTIONAL_SINK_LIMIT,
    T + "heat_dump.constraints.resource": -OPTIONAL_SINK_LIMIT,
    T + "heat_dump.costs.monetary.om_con": HEAT_DUMP_PENALTY,

    # --------------------------------------------------------------------------
    # Battery
    # --------------------------------------------------------------------------
    T + "battery.constraints.energy_eff": BAT_EFF,
    T + "battery.constraints.storage_loss": BAT_LOSS,
    T + "battery.constraints.energy_cap_per_storage_cap_max": BAT_C_RATE,
    T + "battery.constraints.storage_discharge_depth": BAT_MIN_SOC,
    T + "battery.constraints.lifetime": LIFE_BAT,
    T + "battery.costs.monetary.interest_rate": WACC,
    T + "battery.costs.monetary.storage_cap": CAPEX_BAT,
    T + "battery.costs.monetary.om_prod": BAT_OM_PROD,

    # --------------------------------------------------------------------------
    # TES
    # --------------------------------------------------------------------------
    T + "tes.constraints.energy_eff": TES_EFF,
    T + "tes.constraints.storage_loss": TES_LOSS,
    T + "tes.constraints.energy_cap_per_storage_cap_max": TES_C_RATE,
    T + "tes.constraints.lifetime": LIFE_TES,
    T + "tes.costs.monetary.interest_rate": WACC,
    T + "tes.costs.monetary.storage_cap": CAPEX_TES,
    T + "tes.costs.monetary.om_prod": TES_OM_PROD,

    # --------------------------------------------------------------------------
    # Heat pump
    # --------------------------------------------------------------------------
    T + "heat_pump.constraints.energy_eff": COP_HP,
    T + "heat_pump.constraints.energy_cap_max": HP_MAX,
    T + "heat_pump.constraints.energy_ramping": HP_RAMP,
    T + "heat_pump.constraints.lifetime": LIFE_HP,
    T + "heat_pump.costs.monetary.interest_rate": WACC,
    T + "heat_pump.costs.monetary.energy_cap": CAPEX_HP,

    # --------------------------------------------------------------------------
    # CHP (primary output = el_gen; secondary output = heat_chp)
    # --------------------------------------------------------------------------
    T + "chp.constraints.energy_eff": ETA_EL_CHP,
    T + "chp.constraints.carrier_ratios.carrier_out_2.heat_chp": CHP_HEAT_RATIO,
    T + "chp.constraints.energy_cap_max": CHP_MAX_EL,
    T + "chp.constraints.energy_ramping": CHP_RAMP,
    T + "chp.constraints.lifetime": LIFE_CHP,
    T + "chp.costs.monetary.interest_rate": WACC,
    T + "chp.costs.monetary.energy_cap": CAPEX_CHP,

    # --------------------------------------------------------------------------
    # AUX boiler
    # --------------------------------------------------------------------------
    T + "aux_boiler.constraints.energy_eff": AUX_EFF,
    T + "aux_boiler.constraints.energy_cap_max": AUX_MAX,
    T + "aux_boiler.constraints.lifetime": LIFE_AUX,
    T + "aux_boiler.costs.monetary.interest_rate": WACC,
    T + "aux_boiler.costs.monetary.energy_cap": CAPEX_AUX,
    # IMPORTANT: this is a production penalty on AUX heat, not a second gas price.
    T + "aux_boiler.costs.monetary.om_prod": AUX_BACKUP_PENALTY,

    # --------------------------------------------------------------------------
    # Lossless carrier bridges
    # --------------------------------------------------------------------------
    T + "el_wire.constraints.energy_eff": WIRE_EFF,
    T + "heat_chp_wire.constraints.energy_eff": WIRE_EFF,
}

model = calliope.Model(str(MODEL_YAML), override_dict=override_dict)
model.run(build_only=True)

# Backend loc-tech keys used by the custom budget and result extraction
pv_key  = f"{loc}::pv"
bat_key = f"{loc}::battery"
tes_key = f"{loc}::tes"
hp_key  = f"{loc}::heat_pump"
chp_key = f"{loc}::chp"
aux_key = f"{loc}::aux_boiler"


# ==============================================================================
# SECTION 5 — CUSTOM PYOMO CONSTRAINT: RAW OVERNIGHT CAPEX BUDGET
# ==============================================================================

# Calliope's objective annualises investment costs using CRF (WACC + lifetime).
# The thesis budget is different: it is a raw upfront CAPEX ceiling. Therefore
# a custom linear Pyomo constraint is appropriate and preserves exact parity
# with equivalent oemof/PyPSA formulations.
# ==============================================================================

m = model._backend_model


def overnight_budget_rule(backend_model):
    return (
        CAPEX_PV  * backend_model.energy_cap[pv_key]
        + CAPEX_BAT * backend_model.storage_cap[bat_key]
        + CAPEX_TES * backend_model.storage_cap[tes_key]
        + CAPEX_HP  * backend_model.energy_cap[hp_key]
        + CAPEX_CHP * backend_model.energy_cap[chp_key]
        + CAPEX_AUX * backend_model.energy_cap[aux_key]
    ) <= BUDGET


m.overnight_capex_budget = Constraint(rule=overnight_budget_rule)

# ==============================================================================
# SECTION 6 — SOLVE WITH GUROBI
# ==============================================================================

solver = SolverFactory("gurobi_direct")
solve_result = solver.solve(m, tee=True)

if solve_result.solver.status != SolverStatus.ok:
    raise RuntimeError(f"Gurobi solver status is not OK: {solve_result.solver.status}")
if solve_result.solver.termination_condition != TerminationCondition.optimal:
    raise RuntimeError(
        "Optimisation did not terminate optimally: "
        f"{solve_result.solver.termination_condition}"
    )


# ==============================================================================
# SECTION 7 — EXTRACT RESULTS DIRECTLY FROM PYOMO VARIABLES
# ==============================================================================

# Ordered backend timestep keys
TS_LIST = list(m.timesteps)


def get_cap(var, key: str) -> float:
    """Read a scalar Pyomo capacity variable and guard against tiny negatives."""
    try:
        value = var[key].value
        return max(float(value or 0.0), 0.0)
    except (KeyError, TypeError, AttributeError):
        return 0.0


def from_prod(loc_tech_carrier_key: str) -> pd.Series:
    """carrier_prod[loc::tech::carrier,t] → non-negative production series."""
    vals = []
    for ts in TS_LIST:
        try:
            value = m.carrier_prod[loc_tech_carrier_key, ts].value
            vals.append(max(float(value or 0.0), 0.0))
        except (KeyError, TypeError):
            vals.append(0.0)
    return pd.Series(vals, index=data.index)


def from_con(loc_tech_carrier_key: str) -> pd.Series:
    """carrier_con is negative in Calliope; return positive consumption values."""
    vals = []
    for ts in TS_LIST:
        try:
            value = m.carrier_con[loc_tech_carrier_key, ts].value
            vals.append(max(-float(value or 0.0), 0.0))
        except (KeyError, TypeError):
            vals.append(0.0)
    return pd.Series(vals, index=data.index)


def from_storage(loc_tech_key: str) -> pd.Series:
    """storage[loc::tech,t] → state-of-charge series."""
    vals = []
    for ts in TS_LIST:
        try:
            value = m.storage[loc_tech_key, ts].value
            vals.append(float(value or 0.0))
        except (KeyError, TypeError):
            vals.append(0.0)
    return pd.Series(vals, index=data.index)


# Installed capacities
pv_kw     = get_cap(m.energy_cap, pv_key)
bat_kwh   = get_cap(m.storage_cap, bat_key)
bat_kw    = bat_kwh * BAT_C_RATE       # physical C-rate ceiling, not a CAPEX item
tes_kwh   = get_cap(m.storage_cap, tes_key)
tes_kw_th = tes_kwh * TES_C_RATE       # physical C-rate ceiling
hp_kw_th  = get_cap(m.energy_cap, hp_key)
hp_kw_el  = hp_kw_th / COP_HP
chp_kw_el = get_cap(m.energy_cap, chp_key)  # primary carrier_out = el_gen
chp_kw_th = chp_kw_el * CHP_HEAT_RATIO
aux_kw_th = get_cap(m.energy_cap, aux_key)

# Raw CAPEX breakdown used by custom budget
pv_capex    = CAPEX_PV  * pv_kw
bat_capex   = CAPEX_BAT * bat_kwh
tes_capex   = CAPEX_TES * tes_kwh
hp_capex    = CAPEX_HP  * hp_kw_th
chp_capex   = CAPEX_CHP * chp_kw_el
aux_capex   = CAPEX_AUX * aux_kw_th
total_capex = pv_capex + bat_capex + tes_capex + hp_capex + chp_capex + aux_capex

# Hourly dispatch
pv_gen        = from_prod(f"{loc}::pv::el_gen")
grid_imp      = from_prod(f"{loc}::grid_import::electricity")
grid_exp      = from_con(f"{loc}::grid_export::electricity")
chp_el_out    = from_prod(f"{loc}::chp::el_gen")
chp_th_out    = from_prod(f"{loc}::chp::heat_chp")
chp_gas_in    = from_con(f"{loc}::chp::gas")
hp_th_out     = from_prod(f"{loc}::heat_pump::heat")
hp_el_in      = from_con(f"{loc}::heat_pump::electricity")
aux_th_out    = from_prod(f"{loc}::aux_boiler::heat")
aux_gas_in    = from_con(f"{loc}::aux_boiler::gas")
bat_charge    = from_con(f"{loc}::battery::el_gen")
bat_discharge = from_prod(f"{loc}::battery::el_gen")
bat_soc_ts    = from_storage(bat_key)
tes_charge    = from_con(f"{loc}::tes::heat_chp")
tes_discharge = from_prod(f"{loc}::tes::heat_chp")
tes_soc_ts    = from_storage(tes_key)
heat_dump_ts  = from_con(f"{loc}::heat_dump::heat")

# Lossless bridge outputs, used to match the oemof/PyPSA hourly-result structure
onsite_el_direct = from_prod(f"{loc}::el_wire::electricity")
chp_heat_direct  = from_prod(f"{loc}::heat_chp_wire::heat")

# Annual variable-cost components 
grid_imp_cost       = float((grid_imp * data["grid_price_eur_kWh"]).sum())
gas_cost_chp        = float(chp_gas_in.sum()) * GAS_PRICE
gas_cost_aux        = float(aux_gas_in.sum()) * GAS_PRICE
aux_backup_cost     = float(aux_th_out.sum()) * AUX_BACKUP_PENALTY
battery_cycle_cost  = float(bat_discharge.sum()) * BAT_OM_PROD
tes_cycle_cost      = float(tes_discharge.sum()) * TES_OM_PROD
heat_dump_cost      = float(heat_dump_ts.sum()) * HEAT_DUMP_PENALTY
grid_exp_income     = float((grid_exp * data["feedin_tariff_eur_kWh"]).sum())

# Battery SOC validation after optimisation 
if bat_kwh > 1e-9:
    bat_soc_frac = bat_soc_ts / bat_kwh
    bat_soc_min_achieved = float(bat_soc_frac.min())
    bat_soc_max_achieved = float(bat_soc_frac.max())
    if bat_soc_min_achieved < BAT_MIN_SOC - 1e-6:
        raise AssertionError("Solved battery SOC violates BAT_MIN_SOC.")
    if bat_soc_max_achieved > BAT_MAX_SOC + 1e-6:
        raise AssertionError("Solved battery SOC exceeds native 100% upper bound.")
else:
    bat_soc_min_achieved = 0.0
    bat_soc_max_achieved = 0.0

# Bus-balance series for the common hourly-result structure

onsite_el_source_balance = (
    pv_gen + chp_el_out + bat_discharge - bat_charge - onsite_el_direct
)

chp_heat_source_balance = (
    chp_th_out + tes_discharge - tes_charge - chp_heat_direct
)

el_bus_balance = (
    grid_imp + onsite_el_direct
    - data["el_demand_kW"] - hp_el_in - grid_exp
)

heat_bus_balance = (
    chp_heat_direct + hp_th_out + aux_th_out + tes_discharge
    - data["heat_demand_kW"] - heat_dump_ts
)


# ==============================================================================
# SECTION 8 — PRINT SUMMARY
# ==============================================================================

print()
print("=" * 70)
print("CALLIOPE LP — OPTIMAL INVESTMENT RESULTS")
print("=" * 70)
print(f"PV          : {pv_kw:8.2f} kWp")
print(f"Battery     : {bat_kwh:8.2f} kWh   | C-rate ceiling {bat_kw:.2f} kW")
print(f"TES         : {tes_kwh:8.2f} kWh_th | C-rate ceiling {tes_kw_th:.2f} kW_th")
print(f"Heat Pump   : {hp_kw_th:8.2f} kW_th | {hp_kw_el:.2f} kW_el at COP {COP_HP:.2f}")
print(f"CHP         : {chp_kw_el:8.2f} kW_el | {chp_kw_th:.2f} kW_th")
print(f"Aux Boiler  : {aux_kw_th:8.2f} kW_th")
print(f"Total CAPEX : {total_capex:8.2f} EUR  (budget: {BUDGET:,.0f} EUR)")
print(f"Battery SOC min/max achieved : {bat_soc_min_achieved:.4f} / {bat_soc_max_achieved:.4f}")
print("Storage capacity upper bound : none for Battery + TES (budget only)")

print("\n── Raw CAPEX Breakdown ─────────────────────────────────────────────")
print(f"  PV        : {pv_capex:10.0f} EUR")
print(f"  Battery   : {bat_capex:10.0f} EUR")
print(f"  TES       : {tes_capex:10.0f} EUR")
print(f"  Heat Pump : {hp_capex:10.0f} EUR")
print(f"  CHP       : {chp_capex:10.0f} EUR")
print(f"  AUX       : {aux_capex:10.0f} EUR")

print("\n── Annual Variable-Cost Terms ──────────────────────────────────────")
print(f"  Grid import       : {grid_imp_cost:10.0f} EUR/yr")
print(f"  Gas (CHP)         : {gas_cost_chp:10.0f} EUR/yr")
print(f"  Gas (AUX)         : {gas_cost_aux:10.0f} EUR/yr")
print(f"  AUX backup penalty: {aux_backup_cost:10.0f} EUR/yr")
print(f"  Battery cycling   : {battery_cycle_cost:10.0f} EUR/yr")
print(f"  TES cycling       : {tes_cycle_cost:10.0f} EUR/yr")
print(f"  Heat dump         : {heat_dump_cost:10.0f} EUR/yr")
print(f"  Export income     : {grid_exp_income:10.0f} EUR/yr")

print("\n── Annual Energy Totals ────────────────────────────────────────────")
print(f"  Grid import    : {float(grid_imp.sum()):12.0f} kWh/yr")
print(f"  Grid export    : {float(grid_exp.sum()):12.0f} kWh/yr")
print(f"  CHP el output  : {float(chp_el_out.sum()):12.0f} kWh/yr")
print(f"  CHP th output  : {float(chp_th_out.sum()):12.0f} kWh_th/yr")
print(f"  HP th output   : {float(hp_th_out.sum()):12.0f} kWh_th/yr")
print(f"  AUX th output  : {float(aux_th_out.sum()):12.0f} kWh_th/yr")
print(f"  Heat dumped    : {float(heat_dump_ts.sum()):12.0f} kWh_th/yr")


# ==============================================================================
# SECTION 9 — EXPORT TO EXCEL
# ==============================================================================

idx = data.index
out = pd.DataFrame(index=idx)
out.index.name = "Timestamp"

# Exact same hourly column order/names as oemof and PyPSA
out["PV generation (kW)"] = pv_gen.reindex(idx, fill_value=0).values
out["Grid import (kW)"] = grid_imp.reindex(idx, fill_value=0).values
out["Grid export (kW)"] = grid_exp.reindex(idx, fill_value=0).values
out["Onsite electricity direct to main bus (kW)"] = onsite_el_direct.reindex(idx, fill_value=0).values
out["CHP heat direct to main heat bus (kW_th)"] = chp_heat_direct.reindex(idx, fill_value=0).values
out["Heat dump (kW_th)"] = heat_dump_ts.reindex(idx, fill_value=0).values
out["Battery charge (kW)"] = bat_charge.reindex(idx, fill_value=0).values
out["Battery discharge (kW)"] = bat_discharge.reindex(idx, fill_value=0).values
out["Battery SOC (kWh)"] = bat_soc_ts.reindex(idx, fill_value=0).values
out["TES charge (kW)"] = tes_charge.reindex(idx, fill_value=0).values
out["TES discharge (kW)"] = tes_discharge.reindex(idx, fill_value=0).values
out["TES SOC (kWh)"] = tes_soc_ts.reindex(idx, fill_value=0).values
out["Heat pump electricity (kW)"] = hp_el_in.reindex(idx, fill_value=0).values
out["Heat pump heat (kW)"] = hp_th_out.reindex(idx, fill_value=0).values
out["CHP gas input (kW_gas)"] = chp_gas_in.reindex(idx, fill_value=0).values
out["CHP electricity (kW)"] = chp_el_out.reindex(idx, fill_value=0).values
out["CHP heat (kW)"] = chp_th_out.reindex(idx, fill_value=0).values
out["Aux heater gas (kW_gas)"] = aux_gas_in.reindex(idx, fill_value=0).values
out["Aux heater heat (kW)"] = aux_th_out.reindex(idx, fill_value=0).values
out["El demand (kW)"] = data["el_demand_kW"].reindex(idx).values
out["Heat demand (kW)"] = data["heat_demand_kW"].reindex(idx).values
out["Grid price (EUR/kWh)"] = data["grid_price_eur_kWh"].reindex(idx).values
out["Feed-in tariff (EUR/kWh)"] = data["feedin_tariff_eur_kWh"].reindex(idx).values
out["Onsite el source bus balance (kW)"] = onsite_el_source_balance.reindex(idx, fill_value=0).values
out["CHP heat source bus balance (kW_th)"] = chp_heat_source_balance.reindex(idx, fill_value=0).values
out["El bus balance (kW)"] = el_bus_balance.reindex(idx, fill_value=0).values
out["Heat bus balance (kW)"] = heat_bus_balance.reindex(idx, fill_value=0).values

# Same summary-table structure as oemof / PyPSA
summary = pd.DataFrame({
    "Component": [
        "PV",
        "Battery energy capacity",
        "Battery max power",
        "TES",
        "Heat Pump",
        "CHP",
        "Aux Heater",
        "TOTAL CAPEX",
    ],
    "Optimal size": [
        pv_kw,
        bat_kwh,
        bat_kw,
        tes_kwh,
        hp_kw_th,
        chp_kw_el,
        aux_kw_th,
        None,
    ],
    "Unit": [
        "kWp",
        "kWh",
        "kW",
        "kWh_th",
        "kW_th",
        "kW_el",
        "kW_th",
        "",
    ],
    "CAPEX (EUR)": [
        pv_capex,
        bat_capex,
        None,
        tes_capex,
        hp_capex,
        chp_capex,
        aux_capex,
        total_capex,
    ],
})

# Write Summary FIRST and Hourly results SECOND, matching oemof/PyPSA.
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
    out.to_excel(writer, sheet_name="Hourly results")

print(f"\
Results saved → {OUT_XLSX.name}")
