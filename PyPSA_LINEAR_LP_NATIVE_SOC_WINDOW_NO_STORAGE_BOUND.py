"""
================================================================================
PYPSA PURE LINEAR INVESTMENT MODEL (LP)
================================================================================

Rules:
    - Same input data
    - Same components
    - Same technical/economic parameters
    - Same physical assumptions
    - PyPSA-native component formulation
    - Pure LP: no binary/unit-commitment variables

Components:
    PV + Battery + TES + Heat Pump + CHP + Auxiliary Heater
    + Heat Dump + Grid + Gas

Important PyPSA comparison points:
    1. Investment:
         p_nom_extendable=True
       PyPSA's native continuous capacity-expansion mechanism.

    2. Annualized capital cost:
         capital_cost = annuity(WACC, LIFE) * upfront_cost
       Same economic assumptions as oemof, but passed to PyPSA through its
       native capital_cost attribute.

    3. HP/CHP ramping:
         ramp_limit_up / ramp_limit_down
       Native PyPSA Link attributes. No custom ramp constraints.

    4. Battery/TES:
         StorageUnit
       Native p_nom_extendable, max_hours, efficiencies, standing loss,
       cyclic_state_of_charge.

    5. Battery SOC 10%-100%:
       StorageUnit has no native generic fractional min/max SOC attributes.
       With zero battery standing loss, the physical 10%-100% range is represented
       exactly as a 90% usable native StorageUnit energy window. No custom SOC
       constraint is used. it's possible if used Store but have to use more custom constraint

    5b. Storage capacity upper bounds:
       Battery and TES have NO p_nom_max (PyPSA default = unbounded), same as
       oemof. Storage sizing is limited only by costs and the CAPEX budget.

    6. Total upfront CAPEX budget:
       One custom continuous Linopy constraint is required because the budget
       spans Generator + Link + StorageUnit investments.

    7. Battery charging source:
       PV + CHP only, enforced by native bus topology.

    8. TES charging source:
       CHP heat only, enforced by native bus topology.

Solver:
    Gurobi

Output:
    Hourly-results columns are kept identical oemof result workbook.
================================================================================
"""

import pandas as pd
import pypsa
from pypsa.costs import annuity


# =============================================================================
# PART 1 - INPUT DATA
# =============================================================================

data = pd.read_csv("input_data_office_medium.csv",
    index_col=0,
    parse_dates=True,
)

print("PyPSA version :", pypsa.__version__)
print("Input loaded")
print("Timesteps     :", len(data))
print("Columns       :", list(data.columns))
print()


# =============================================================================
# PART 2 - PARAMETERS
# =============================================================================

WACC = 0.04
BUDGET_EUR = 300000   

# PV
# PV  [Fraunhofer ISE PV Report 2025: 10-100 kWp rooftop < 1260 EUR/kWp]
CAPEX_PV = 1100       # EUR/kWp
LIFE_PV = 25
# ---- PV AREA CONSTRAINT (office) ----
# Office building (~1600 m2 floor). Roof-to-floor ratio for a low-rise

NET_USABLE_AREA_M2 = 1600
PV_AREA_RATIO_M2_PER_M2_LIVING = 0.40
PV_AREA_PER_KWP_M2 = 6.0

PV_USABLE_AREA_M2 = (
    NET_USABLE_AREA_M2
    * PV_AREA_RATIO_M2_PER_M2_LIVING
)

PV_MAX_KW = (
    PV_USABLE_AREA_M2
    / PV_AREA_PER_KWP_M2
)

# Building demand reference
PEAK_HEAT_KW = float(data["heat_demand_kW"].max())
PEAK_EL_KW = float(data["el_demand_kW"].max())

# Battery
# Battery  [SolarPower Europe / German C&I 2025]
CAPEX_BAT = 450       # EUR/kWh
LIFE_BAT = 15
BAT_EFF_CHARGE = 0.95
BAT_EFF_DISCHARGE = 0.95
BAT_MIN_SOC = 0.10      # 10% minimum
BAT_MAX_SOC = 1         # 100% maximum
BAT_C_RATE = 1.0

BAT_MAX_HOURS = 1.0 / BAT_C_RATE

# PyPSA StorageUnit has no direct native min/max SOC fraction inputs.
# With zero battery standing loss, the physical 10%-100% operating window is
# represented natively as the usable 90% energy range (max_hours = 0.90 h):
#
# physical SOC = 10% reserve + PyPSA usable SOC
#
# The physical battery is still costed and reported at BAT_MAX_HOURS = 1 h.
BAT_USABLE_SOC_FRACTION = BAT_MAX_SOC - BAT_MIN_SOC
BAT_STORAGE_MAX_HOURS = BAT_MAX_HOURS * BAT_USABLE_SOC_FRACTION

# TES
CAPEX_TES = 50.0        # EUR/kWh_th
LIFE_TES = 20
TES_EFF_CHARGE = 0.98
TES_EFF_DISCHARGE = 0.98
TES_LOSS = 0.001
TES_C_RATE = 0.25

TES_MAX_HOURS = 1.0 / TES_C_RATE

# Heat Pump
CAPEX_HP = 1200.0       # EUR/kW_th
LIFE_HP = 20
COP_HP = 3.0
HP_MAX_KW_TH = PEAK_HEAT_KW
HP_RAMP_UP = 0.50
HP_RAMP_DOWN = 0.50

# CHP
CAPEX_CHP = 1500.0      # EUR/kW_el
LIFE_CHP = 20
ETA_EL_CHP = 0.35
ETA_TH_CHP = 0.45
CHP_RAMP_UP = 0.25
CHP_RAMP_DOWN = 0.25

# CHP thermal maximum linked to building peak heat demand.
CHP_MAX_KW_TH = PEAK_HEAT_KW
CHP_MAX_KW_GAS = CHP_MAX_KW_TH / ETA_TH_CHP
CHP_MAX_KW_EL = CHP_MAX_KW_GAS * ETA_EL_CHP

# Auxiliary Heater
CAPEX_AUX = 150       # EUR/kW_th
LIFE_AUX = 20
AUX_EFF = 0.90
AUX_MAX_KW_TH = PEAK_HEAT_KW

# Storage upper bounds
# Battery and TES: NO upper bound (PyPSA default p_nom_max = inf), same as oemof.
# Sizing is limited only by investment cost and the upfront CAPEX budget.

# Operating costs
GAS_PRICE_EUR_KWH = 0.07       #[Eurostat non-household DE 2024]

BAT_DISCHARGE_COST_EUR_KWH = 0.001
TES_DISCHARGE_COST_EUR_KWH = 0.001
HEAT_DUMP_COST_EUR_KWH_TH = 0.0001

# Aux remains fully optional, but expensive enough to act as backup.
HP_WORST_CASE_HEAT_COST = (
    float(data["grid_price_eur_kWh"].max())
    / COP_HP
)

AUX_BASE_HEAT_COST = (
    GAS_PRICE_EUR_KWH
    / AUX_EFF
)

AUX_BACKUP_PENALTY = max(
    0.0,
    HP_WORST_CASE_HEAT_COST
    - AUX_BASE_HEAT_COST,
) + 0.01


# =============================================================================
# PART 3 - ANNUALISED CAPITAL COSTS
# =============================================================================
# PyPSA investment variables are in MW.
# Therefore all capital_cost values below are EUR/MW/year.

# PV:
# p_nom reference = MW electrical PV capacity.
PV_CAPITAL_COST = (
    annuity(WACC, LIFE_PV)
    * CAPEX_PV
    * 1000.0
)

# Battery:
# StorageUnit p_nom reference = MW power.
# Energy capacity = p_nom * max_hours.
BAT_CAPITAL_COST = (
    annuity(WACC, LIFE_BAT)
    * CAPEX_BAT
    * 1000.0
    * BAT_MAX_HOURS
)

# TES:
# StorageUnit p_nom reference = MW thermal power.
# Energy capacity = p_nom * max_hours.
TES_CAPITAL_COST = (
    annuity(WACC, LIFE_TES)
    * CAPEX_TES
    * 1000.0
    * TES_MAX_HOURS
)

# HP:
# Link p_nom reference = MW electrical INPUT.
# Thermal capacity = electrical input * COP.
HP_CAPITAL_COST = (
    annuity(WACC, LIFE_HP)
    * CAPEX_HP
    * COP_HP
    * 1000.0
)

# CHP:
# Link p_nom reference = MW gas INPUT.
# Electrical installed capacity = gas input * ETA_EL_CHP.
CHP_CAPITAL_COST = (
    annuity(WACC, LIFE_CHP)
    * CAPEX_CHP
    * ETA_EL_CHP
    * 1000.0
)

# Aux:
# Link p_nom reference = MW gas INPUT.
# Thermal capacity = gas input * AUX_EFF.
AUX_CAPITAL_COST = (
    annuity(WACC, LIFE_AUX)
    * CAPEX_AUX
    * AUX_EFF
    * 1000.0
)

print("Annualized capital costs:")
print(f"  PV         : {PV_CAPITAL_COST:,.2f} EUR/MW/year")
print(f"  Battery    : {BAT_CAPITAL_COST:,.2f} EUR/MW_power/year")
print(f"  TES        : {TES_CAPITAL_COST:,.2f} EUR/MW_th_power/year")
print(f"  Heat Pump  : {HP_CAPITAL_COST:,.2f} EUR/MW_el_input/year")
print(f"  CHP        : {CHP_CAPITAL_COST:,.2f} EUR/MW_gas_input/year")
print(f"  Aux Heater : {AUX_CAPITAL_COST:,.2f} EUR/MW_gas_input/year")
print()


# =============================================================================
# PART 4 - UPFRONT COST REFERENCES FOR THE BUDGET
# =============================================================================
# These are NOT objective capital_cost values.
# They are only the raw upfront CAPEX coefficients used in the
# investment-budget constraint (BUDGET_EUR).

PV_UPFRONT_COST_EUR_MW = (
    CAPEX_PV
    * 1000.0
)

BAT_UPFRONT_COST_EUR_MW = (
    CAPEX_BAT
    * 1000.0
    * BAT_MAX_HOURS
)

TES_UPFRONT_COST_EUR_MW = (
    CAPEX_TES
    * 1000.0
    * TES_MAX_HOURS
)

HP_UPFRONT_COST_EUR_MW = (
    CAPEX_HP
    * COP_HP
    * 1000.0
)

CHP_UPFRONT_COST_EUR_MW = (
    CAPEX_CHP
    * ETA_EL_CHP
    * 1000.0
)

AUX_UPFRONT_COST_EUR_MW = (
    CAPEX_AUX
    * AUX_EFF
    * 1000.0
)


# =============================================================================
# PART 5 - UNIT CONVERSIONS
# =============================================================================

el_demand_mw = (
    data["el_demand_kW"]
    / 1000.0
)

heat_demand_mw = (
    data["heat_demand_kW"]
    / 1000.0
)

grid_price_eur_mwh = (
    data["grid_price_eur_kWh"]
    * 1000.0
)

feedin_tariff_eur_mwh = (
    data["feedin_tariff_eur_kWh"]
    * 1000.0
)

gas_price_eur_mwh = (
    GAS_PRICE_EUR_KWH
    * 1000.0
)

bat_discharge_cost_eur_mwh = (
    BAT_DISCHARGE_COST_EUR_KWH
    * 1000.0
)

tes_discharge_cost_eur_mwh = (
    TES_DISCHARGE_COST_EUR_KWH
    * 1000.0
)

heat_dump_cost_eur_mwh = (
    HEAT_DUMP_COST_EUR_KWH_TH
    * 1000.0
)

# Link marginal_cost is charged on p0 input.
aux_backup_penalty_eur_mwh_gas = (
    AUX_BACKUP_PENALTY
    * 1000.0
    * AUX_EFF
)


# =============================================================================
# PART 6 - NETWORK + BUSES
# =============================================================================

n = pypsa.Network()
n.set_snapshots(data.index)

n.add("Bus", "electricity")
n.add("Bus", "heat")
n.add("Bus", "gas")

# Dedicated source buses.
n.add("Bus", "onsite_electricity")
n.add("Bus", "chp_heat")


# =============================================================================
# PART 7 - LOADS
# =============================================================================

n.add(
    "Load",
    "electricity_demand",
    bus="electricity",
    p_set=el_demand_mw,
)

n.add(
    "Load",
    "heat_demand",
    bus="heat",
    p_set=heat_demand_mw,
)


# =============================================================================
# PART 8 - GRID + GAS
# =============================================================================

GRID_LIMIT_MW = 1.0
GAS_LIMIT_MW = 1.0
HEAT_DUMP_LIMIT_MW = 1.0

n.add(
    "Generator",
    "grid_import",
    bus="electricity",
    p_nom=GRID_LIMIT_MW,
    marginal_cost=grid_price_eur_mwh,
)

n.add(
    "Generator",
    "grid_export",
    bus="electricity",
    sign=-1,
    p_nom=GRID_LIMIT_MW,
    marginal_cost=-feedin_tariff_eur_mwh,
)

n.add(
    "Generator",
    "gas_supply",
    bus="gas",
    p_nom=GAS_LIMIT_MW,
    marginal_cost=gas_price_eur_mwh,
)


# =============================================================================
# PART 9 - PV INVESTMENT
# =============================================================================
# PyPSA-native continuous investment:
#     p_nom_extendable=True
#
# capital_cost contains the annualised investment cost calculated above.

n.add(
    "Generator",
    "pv",
    bus="onsite_electricity",
    p_nom_extendable=True,
    p_nom_min=0.0,
    p_nom_max=PV_MAX_KW / 1000.0,
    p_min_pu=data["pv_cf"],
    p_max_pu=data["pv_cf"],
    capital_cost=PV_CAPITAL_COST,
    marginal_cost=0.0,
)


# =============================================================================
# PART 10 - ONSITE ELECTRICITY TRANSFER
# =============================================================================
# PV and CHP electricity enter onsite_electricity.
# Battery also uses this bus.
# Grid is on the main electricity bus, so grid cannot charge Battery.
#
# Lossless bridge. Its capacity must NOT act as a hidden battery limit
# (battery is unbounded), so it is set to a large non-binding value,
# equivalent to the unbounded oemof Converter / Calliope el_wire.

BRIDGE_LIMIT_MW = 1.0     # 1 MW, non-binding (same as grid/gas limits)

n.add(
    "Link",
    "onsite_el_to_main",
    bus0="onsite_electricity",
    bus1="electricity",
    p_nom=BRIDGE_LIMIT_MW,
    efficiency=1.0,
    p_min_pu=0.0,
)


# =============================================================================
# PART 11 - BATTERY INVESTMENT
# =============================================================================
# StorageUnit chosen deliberately for the thesis tool comparison.
# Direct min/max SOC fractions are not available natively in StorageUnit.
# Therefore the native state variable represents only the usable 90% window
# (physical 10%-100%). No p_nom_max -> unbounded (PyPSA default).

n.add(
    "StorageUnit",
    "battery",
    bus="onsite_electricity",
    p_nom_extendable=True,
    p_nom_min=0.0,
    max_hours=BAT_STORAGE_MAX_HOURS,
    efficiency_store=BAT_EFF_CHARGE,
    efficiency_dispatch=BAT_EFF_DISCHARGE,
    standing_loss=0.0,
    cyclic_state_of_charge=True,
    capital_cost=BAT_CAPITAL_COST,
    marginal_cost=bat_discharge_cost_eur_mwh,
)


# =============================================================================
# PART 12 - CHP INVESTMENT + NATIVE RAMPING
# =============================================================================

n.add(
    "Link",
    "chp",
    bus0="gas",
    bus1="onsite_electricity",
    bus2="chp_heat",
    efficiency=ETA_EL_CHP,
    efficiency2=ETA_TH_CHP,
    p_nom_extendable=True,
    p_nom_min=0.0,
    p_nom_max=CHP_MAX_KW_GAS / 1000.0,
    capital_cost=CHP_CAPITAL_COST,
    marginal_cost=0.0,
    ramp_limit_up=CHP_RAMP_UP,
    ramp_limit_down=CHP_RAMP_DOWN,
)


# =============================================================================
# PART 13 - CHP HEAT TRANSFER
# =============================================================================
# Lossless bridge, non-binding (TES is unbounded).

n.add(
    "Link",
    "chp_heat_to_main",
    bus0="chp_heat",
    bus1="heat",
    p_nom=BRIDGE_LIMIT_MW,
    efficiency=1.0,
    p_min_pu=0.0,
)


# =============================================================================
# PART 14 - TES INVESTMENT
# =============================================================================
# TES can charge only from the CHP heat bus.
# No p_nom_max , unbounded (PyPSA default), same as oemof.

n.add(
    "StorageUnit",
    "tes",
    bus="chp_heat",
    p_nom_extendable=True,
    p_nom_min=0.0,
    max_hours=TES_MAX_HOURS,
    efficiency_store=TES_EFF_CHARGE,
    efficiency_dispatch=TES_EFF_DISCHARGE,
    standing_loss=TES_LOSS,
    cyclic_state_of_charge=True,
    capital_cost=TES_CAPITAL_COST,
    marginal_cost=tes_discharge_cost_eur_mwh,
)


# =============================================================================
# PART 15 - HEAT PUMP INVESTMENT + NATIVE RAMPING
# =============================================================================

n.add(
    "Link",
    "heat_pump",
    bus0="electricity",
    bus1="heat",
    efficiency=COP_HP,
    p_nom_extendable=True,
    p_nom_min=0.0,
    p_nom_max=(HP_MAX_KW_TH / COP_HP) / 1000.0,
    capital_cost=HP_CAPITAL_COST,
    marginal_cost=0.0,
    ramp_limit_up=HP_RAMP_UP,
    ramp_limit_down=HP_RAMP_DOWN,
)


# =============================================================================
# PART 16 - AUXILIARY HEATER INVESTMENT
# =============================================================================
# Fully optimized from zero.
# No fixed minimum capacity.

n.add(
    "Link",
    "aux_heater",
    bus0="gas",
    bus1="heat",
    efficiency=AUX_EFF,
    p_nom_extendable=True,
    p_nom_min=0.0,
    p_nom_max=(AUX_MAX_KW_TH / AUX_EFF) / 1000.0,
    capital_cost=AUX_CAPITAL_COST,
    marginal_cost=aux_backup_penalty_eur_mwh_gas,
)


# =============================================================================
# PART 17 - HEAT DUMP
# =============================================================================

n.add(
    "Generator",
    "heat_dump",
    bus="heat",
    sign=-1,
    p_nom=HEAT_DUMP_LIMIT_MW,
    marginal_cost=heat_dump_cost_eur_mwh,
)


# =============================================================================
# PART 18 - CREATE PYPSA OPTIMIZATION MODEL
# =============================================================================

n.optimize.create_model(include_objective_constant=False)

m = n.model


# =============================================================================
# PART 19 - CUSTOM LINEAR UPFRONT CAPEX BUDGET
# =============================================================================

total_upfront_capex = (

      m["Generator-p_nom"].at["pv"]
      * PV_UPFRONT_COST_EUR_MW

    + m["StorageUnit-p_nom"].at["battery"]
      * BAT_UPFRONT_COST_EUR_MW

    + m["StorageUnit-p_nom"].at["tes"]
      * TES_UPFRONT_COST_EUR_MW

    + m["Link-p_nom"].at["heat_pump"]
      * HP_UPFRONT_COST_EUR_MW

    + m["Link-p_nom"].at["chp"]
      * CHP_UPFRONT_COST_EUR_MW

    + m["Link-p_nom"].at["aux_heater"]
      * AUX_UPFRONT_COST_EUR_MW
)

m.add_constraints(
    total_upfront_capex <= BUDGET_EUR,
    name="Total-upfront-CAPEX-budget",
)


# =============================================================================
# PART 20 - SOLVE
# =============================================================================

SOLVER = "gurobi"

print("=" * 76)
print("SOLVING PYPSA PURE LP")
print("=" * 76)
print("Investment             : native p_nom_extendable")
print("Annualized CAPEX       : annuity() -> native capital_cost")
print("HP/CHP ramp            : native Link.ramp_limit_up/down")
print("Battery/TES            : native StorageUnit")
print("Custom constraints     : upfront CAPEX budget only")
print("Solver                 :", SOLVER.upper())
print("=" * 76)

status, condition = n.optimize.solve_model(
    solver_name=SOLVER,
)

print()
print("Solver status    :", status)
print("Solver condition :", condition)
print()

if status != "ok":

    raise RuntimeError(
        f"Optimization failed: "
        f"{status}, {condition}"
    )


# =============================================================================
# PART 21 - OPTIMIZED CAPACITIES
# =============================================================================

pv_kw = (
    n.generators.at[
        "pv",
        "p_nom_opt",
    ]
    * 1000.0
)

battery_kw = (
    n.storage_units.at[
        "battery",
        "p_nom_opt",
    ]
    * 1000.0
)

battery_kwh = (
    battery_kw
    * BAT_MAX_HOURS
)

tes_kw_th = (
    n.storage_units.at[
        "tes",
        "p_nom_opt",
    ]
    * 1000.0
)

tes_kwh_th = (
    tes_kw_th
    * TES_MAX_HOURS
)

hp_kw_el = (
    n.links.at[
        "heat_pump",
        "p_nom_opt",
    ]
    * 1000.0
)

hp_kw_th = (
    hp_kw_el
    * COP_HP
)

chp_kw_gas = (
    n.links.at[
        "chp",
        "p_nom_opt",
    ]
    * 1000.0
)

chp_kw_el = (
    chp_kw_gas
    * ETA_EL_CHP
)

chp_kw_th = (
    chp_kw_gas
    * ETA_TH_CHP
)

aux_kw_gas = (
    n.links.at[
        "aux_heater",
        "p_nom_opt",
    ]
    * 1000.0
)

aux_kw_th = (
    aux_kw_gas
    * AUX_EFF
)


# =============================================================================
# PART 22 - UPFRONT CAPEX RESULT
# =============================================================================

pv_capex = (
    CAPEX_PV
    * pv_kw
)

battery_capex = (
    CAPEX_BAT
    * battery_kwh
)

tes_capex = (
    CAPEX_TES
    * tes_kwh_th
)

hp_capex = (
    CAPEX_HP
    * hp_kw_th
)

chp_capex = (
    CAPEX_CHP
    * chp_kw_el
)

aux_capex = (
    CAPEX_AUX
    * aux_kw_th
)

total_capex = (
    pv_capex
    + battery_capex
    + tes_capex
    + hp_capex
    + chp_capex
    + aux_capex
)


# =============================================================================
# PART 23 - HOURLY RESULTS
# =============================================================================

pv_gen = (
    n.generators_t.p["pv"]
    * 1000.0
)

grid_imp = (
    n.generators_t.p["grid_import"]
    * 1000.0
)

grid_exp = (
    n.generators_t.p["grid_export"]
    * 1000.0
)

heat_dump_flow = (
    n.generators_t.p["heat_dump"]
    * 1000.0
)

onsite_el_direct = (
    -n.links_t.p1[
        "onsite_el_to_main"
    ]
    * 1000.0
)

chp_heat_direct = (
    -n.links_t.p1[
        "chp_heat_to_main"
    ]
    * 1000.0
)

bat_charge = (
    n.storage_units_t.p_store[
        "battery"
    ]
    * 1000.0
)

bat_discharge = (
    n.storage_units_t.p_dispatch[
        "battery"
    ]
    * 1000.0
)

bat_soc_usable = (
    n.storage_units_t.state_of_charge[
        "battery"
    ]
    * 1000.0
)

# Reconstruct physical battery SOC:
# 0 usable SOC corresponds to 10% physical SOC.
bat_soc = (
    bat_soc_usable
    + BAT_MIN_SOC * battery_kwh
)

tes_charge = (
    n.storage_units_t.p_store[
        "tes"
    ]
    * 1000.0
)

tes_discharge = (
    n.storage_units_t.p_dispatch[
        "tes"
    ]
    * 1000.0
)

tes_soc = (
    n.storage_units_t.state_of_charge[
        "tes"
    ]
    * 1000.0
)

hp_el = (
    n.links_t.p0[
        "heat_pump"
    ]
    * 1000.0
)

hp_heat = (
    -n.links_t.p1[
        "heat_pump"
    ]
    * 1000.0
)

chp_gas = (
    n.links_t.p0[
        "chp"
    ]
    * 1000.0
)

chp_el = (
    -n.links_t.p1[
        "chp"
    ]
    * 1000.0
)

chp_heat = (
    -n.links_t.p2[
        "chp"
    ]
    * 1000.0
)

aux_gas = (
    n.links_t.p0[
        "aux_heater"
    ]
    * 1000.0
)

aux_heat = (
    -n.links_t.p1[
        "aux_heater"
    ]
    * 1000.0
)


# =============================================================================
# PART 24 - EXCEL HOURLY STRUCTURE
# =============================================================================

out = pd.DataFrame(
    index=n.snapshots
)

out["PV generation (kW)"] = pv_gen
out["Grid import (kW)"] = grid_imp
out["Grid export (kW)"] = grid_exp

out[
    "Onsite electricity direct to main bus (kW)"
] = onsite_el_direct

out[
    "CHP heat direct to main heat bus (kW_th)"
] = chp_heat_direct

out["Heat dump (kW_th)"] = heat_dump_flow

out["Battery charge (kW)"] = bat_charge
out["Battery discharge (kW)"] = bat_discharge
out["Battery SOC (kWh)"] = bat_soc

out["TES charge (kW)"] = tes_charge
out["TES discharge (kW)"] = tes_discharge
out["TES SOC (kWh)"] = tes_soc

out["Heat pump electricity (kW)"] = hp_el
out["Heat pump heat (kW)"] = hp_heat

out["CHP gas input (kW_gas)"] = chp_gas
out["CHP electricity (kW)"] = chp_el
out["CHP heat (kW)"] = chp_heat

out["Aux heater gas (kW_gas)"] = aux_gas
out["Aux heater heat (kW)"] = aux_heat

out["El demand (kW)"] = (
    data["el_demand_kW"]
    .reindex(out.index)
)

out["Heat demand (kW)"] = (
    data["heat_demand_kW"]
    .reindex(out.index)
)

out["Grid price (EUR/kWh)"] = (
    data["grid_price_eur_kWh"]
    .reindex(out.index)
)

out["Feed-in tariff (EUR/kWh)"] = (
    data["feedin_tariff_eur_kWh"]
    .reindex(out.index)
)


# =============================================================================
# PART 25 - BUS BALANCE VALIDATION
# =============================================================================

out[
    "Onsite el source bus balance (kW)"
] = (

    pv_gen
    + chp_el
    + bat_discharge
    - bat_charge
    - onsite_el_direct
)

out[
    "CHP heat source bus balance (kW_th)"
] = (

    chp_heat
    + tes_discharge
    - tes_charge
    - chp_heat_direct
)

out[
    "El bus balance (kW)"
] = (

    grid_imp
    + onsite_el_direct
    - out["El demand (kW)"]
    - hp_el
    - grid_exp
)

out[
    "Heat bus balance (kW)"
] = (

    chp_heat_direct
    + hp_heat
    + aux_heat
    - out["Heat demand (kW)"]
    - heat_dump_flow
)


# =============================================================================
# PART 26 - OPERATION VALIDATION
# =============================================================================

FLOW_TOL_KW = 1e-6

battery_simultaneous_hours = int(
    (
        (bat_charge > FLOW_TOL_KW)
        &
        (bat_discharge > FLOW_TOL_KW)
    ).sum()
)

tes_simultaneous_hours = int(
    (
        (tes_charge > FLOW_TOL_KW)
        &
        (tes_discharge > FLOW_TOL_KW)
    ).sum()
)

if battery_kwh > 1e-12:

    bat_soc_fraction = (
        bat_soc
        / battery_kwh
    )

    bat_soc_min_achieved = float(
        bat_soc_fraction.min()
    )

    bat_soc_max_achieved = float(
        bat_soc_fraction.max()
    )

else:

    bat_soc_min_achieved = 0.0
    bat_soc_max_achieved = 0.0


# Native HP ramp validation.
hp_allowed_ramp = (
    HP_RAMP_UP
    * hp_kw_th
)

hp_diff = (
    hp_heat
    .diff()
    .dropna()
)

hp_ramp_violations = int(
    (
        (hp_diff > hp_allowed_ramp + FLOW_TOL_KW)
        |
        (-hp_diff > hp_allowed_ramp + FLOW_TOL_KW)
    ).sum()
)


# Native CHP ramp validation.
chp_allowed_ramp = (
    CHP_RAMP_UP
    * chp_kw_el
)

chp_diff = (
    chp_el
    .diff()
    .dropna()
)

chp_ramp_violations = int(
    (
        (chp_diff > chp_allowed_ramp + FLOW_TOL_KW)
        |
        (-chp_diff > chp_allowed_ramp + FLOW_TOL_KW)
    ).sum()
)


# =============================================================================
# PART 27 - COST BREAKDOWN
# =============================================================================

annualized_capex = (

      n.generators.at[
          "pv",
          "p_nom_opt",
      ]
      * PV_CAPITAL_COST

    + n.storage_units.at[
          "battery",
          "p_nom_opt",
      ]
      * BAT_CAPITAL_COST

    + n.storage_units.at[
          "tes",
          "p_nom_opt",
      ]
      * TES_CAPITAL_COST

    + n.links.at[
          "heat_pump",
          "p_nom_opt",
      ]
      * HP_CAPITAL_COST

    + n.links.at[
          "chp",
          "p_nom_opt",
      ]
      * CHP_CAPITAL_COST

    + n.links.at[
          "aux_heater",
          "p_nom_opt",
      ]
      * AUX_CAPITAL_COST
)

grid_import_cost = float(
    (
        n.generators_t.p[
            "grid_import"
        ]
        * grid_price_eur_mwh
    ).sum()
)

grid_export_revenue = float(
    (
        n.generators_t.p[
            "grid_export"
        ]
        * feedin_tariff_eur_mwh
    ).sum()
)

gas_cost = float(
    (
        n.generators_t.p[
            "gas_supply"
        ]
        * gas_price_eur_mwh
    ).sum()
)

battery_variable_cost = float(
    (
        n.storage_units_t.p_dispatch[
            "battery"
        ]
        * bat_discharge_cost_eur_mwh
    ).sum()
)

tes_variable_cost = float(
    (
        n.storage_units_t.p_dispatch[
            "tes"
        ]
        * tes_discharge_cost_eur_mwh
    ).sum()
)

aux_backup_cost = float(
    (
        n.links_t.p0[
            "aux_heater"
        ]
        * aux_backup_penalty_eur_mwh_gas
    ).sum()
)

heat_dump_cost = float(
    (
        n.generators_t.p[
            "heat_dump"
        ]
        * heat_dump_cost_eur_mwh
    ).sum()
)

calculated_objective = (

    annualized_capex
    + grid_import_cost
    + gas_cost
    + battery_variable_cost
    + tes_variable_cost
    + aux_backup_cost
    + heat_dump_cost
    - grid_export_revenue
)


# =============================================================================
# PART 28 - PRINT RESULTS
# =============================================================================

print()
print("=" * 76)
print("PYPSA PURE LP - OPTIMAL INVESTMENT")
print("=" * 76)

print(
    f"PV          : "
    f"{pv_kw:10.2f} kWp "
    f"(area max: {PV_MAX_KW:.2f} kWp)"
)

print(
    f"Battery     : "
    f"{battery_kwh:10.2f} kWh energy | "
    f"{battery_kw:10.2f} kW max power"
)

print(
    f"TES         : "
    f"{tes_kwh_th:10.2f} kWh_th"
)

print(
    f"Heat Pump   : "
    f"{hp_kw_th:10.2f} kW_th"
)

print(
    f"CHP         : "
    f"{chp_kw_el:10.2f} kW_el"
)

print(
    f"Aux Heater  : "
    f"{aux_kw_th:10.2f} kW_th"
)

print(
    f"Total CAPEX : "
    f"{total_capex:10.2f} EUR / "
    f"{BUDGET_EUR:.2f} EUR"
)

print()

print("Validation")

print(
    "Battery simultaneous charge/discharge h :",
    battery_simultaneous_hours,
)

print(
    "TES simultaneous charge/discharge h     :",
    tes_simultaneous_hours,
)

print(
    "Battery SOC min/max achieved             :",
    f"{bat_soc_min_achieved:.4f}",
    "/",
    f"{bat_soc_max_achieved:.4f}",
)

print(
    "HP native ramp violations                :",
    hp_ramp_violations,
)

print(
    "CHP native ramp violations               :",
    chp_ramp_violations,
)

print(
    "Annual heat dump (kWh_th)                :",
    float(heat_dump_flow.sum()),
)

print(
    "El bus max imbalance (kW)                :",
    float(
        out[
            "El bus balance (kW)"
        ]
        .abs()
        .max()
    ),
)

print(
    "Heat bus max imbalance (kW)              :",
    float(
        out[
            "Heat bus balance (kW)"
        ]
        .abs()
        .max()
    ),
)

print()

print("COST BREAKDOWN")

print(
    f"Annualized CAPEX      : "
    f"{annualized_capex:,.2f} EUR/year"
)

print(
    f"Grid import cost      : "
    f"{grid_import_cost:,.2f} EUR/year"
)

print(
    f"Gas cost              : "
    f"{gas_cost:,.2f} EUR/year"
)

print(
    f"Export revenue        : "
    f"{-grid_export_revenue:,.2f} EUR/year"
)

print(
    f"Calculated objective  : "
    f"{calculated_objective:,.6f} EUR/year"
)

print(
    f"PyPSA objective       : "
    f"{float(n.objective):,.6f} EUR/year"
)

print(
    f"Objective difference  : "
    f"{calculated_objective - float(n.objective):,.8f} EUR"
)


# =============================================================================
# PART 29 - SUMMARY
# =============================================================================

summary = pd.DataFrame(
    {
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
            battery_kwh,
            battery_kw,
            tes_kwh_th,
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
            battery_capex,
            None,
            tes_capex,
            hp_capex,
            chp_capex,
            aux_capex,
            total_capex,
        ],
    }
)

summary_metrics = pd.DataFrame(
    {
        "Metric": [
            "PyPSA version",
            "Solver",
            "Objective (EUR/year)",
            "Calculated objective (EUR/year)",
            "Budget (EUR)",
            "Budget remaining (EUR)",
            "Annualized CAPEX (EUR/year)",
            "PV area maximum (kWp)",
            "PV optimized capacity (kWp)",
            "Storage capacity upper bound",
            "Battery min SOC setting",
            "Battery max SOC setting",
            "HP maximum capacity bound (kW_th)",
            "CHP maximum thermal bound (kW_th)",
            "CHP maximum electrical bound (kW_el)",
            "Aux maximum capacity bound (kW_th)",
            "Battery SOC implementation",
            "HP ramp implementation",
            "CHP ramp implementation",
            "Investment implementation",
            "Annualized capital cost implementation",
            "Upfront budget implementation",
            "Battery simultaneous charge/discharge hours",
            "TES simultaneous charge/discharge hours",
            "Battery SOC min achieved",
            "Battery SOC max achieved",
            "HP ramp violations",
            "CHP ramp violations",
            "Annual heat dump (kWh_th)",
            "Grid import cost (EUR/year)",
            "Gas cost (EUR/year)",
            "Grid export revenue (EUR/year)",
            "El bus max imbalance (kW)",
            "Heat bus max imbalance (kW)",
            "Onsite electricity source-bus max imbalance (kW)",
            "CHP heat source-bus max imbalance (kW_th)",
        ],
        "Value": [
            pypsa.__version__,
            SOLVER.upper(),
            float(n.objective),
            calculated_objective,
            BUDGET_EUR,
            BUDGET_EUR - total_capex,
            annualized_capex,
            PV_MAX_KW,
            pv_kw,
            "None (Battery + TES unbounded, budget only)",
            BAT_MIN_SOC,
            BAT_MAX_SOC,
            HP_MAX_KW_TH,
            CHP_MAX_KW_TH,
            CHP_MAX_KW_EL,
            AUX_MAX_KW_TH,
            "StorageUnit usable 90% SOC window (10%-100%); direct min/max fraction not native",
            "NATIVE Link ramp_limit_up/down",
            "NATIVE Link ramp_limit_up/down",
            "NATIVE p_nom_extendable=True",
            "annuity() -> NATIVE capital_cost",
            "CUSTOM Linopy",
            battery_simultaneous_hours,
            tes_simultaneous_hours,
            bat_soc_min_achieved,
            bat_soc_max_achieved,
            hp_ramp_violations,
            chp_ramp_violations,
            float(heat_dump_flow.sum()),
            grid_import_cost,
            gas_cost,
            grid_export_revenue,
            float(
                out[
                    "El bus balance (kW)"
                ]
                .abs()
                .max()
            ),
            float(
                out[
                    "Heat bus balance (kW)"
                ]
                .abs()
                .max()
            ),
            float(
                out[
                    "Onsite el source bus balance (kW)"
                ]
                .abs()
                .max()
            ),
            float(
                out[
                    "CHP heat source bus balance (kW_th)"
                ]
                .abs()
                .max()
            ),
        ],
    }
)


# =============================================================================
# PART 30 - EXCEL EXPORT
# =============================================================================

out.index.name = "Timestamp"

result_file = (
    "PyPSA_LINEAR_LP_FINAL_NATIVE_SOC_WINDOW__No_Storage_Bound_results.xlsx"
)

with pd.ExcelWriter(
    result_file
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
        startrow=0,
    )

    summary_metrics.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
        startrow=len(summary) + 3,
    )

    out.to_excel(
        writer,
        sheet_name="Hourly results",
    )

print()
print("Saved:", result_file)
print("Finished.")
