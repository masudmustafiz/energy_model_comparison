"""
================================================================================
OEMOF.SOLPH LINEAR INVESTMENT MODEL (LP)
================================================================================
Components kept:
    PV + Battery + TES + Heat Pump + CHP + Auxiliary Heater + Heat Dump + Grid + Gas

What makes this model LINEAR:
    - NO solph.NonConvex(...)
    - NO binary on/off variables
    - NO startup/shutdown integer logic
    - NO minimum up/down-time logic
    - NO binary charge/discharge mode

What is still optimized:
    - Continuous investment capacities
    - Hourly dispatch
    - Generic Battery/TES state of charge
    - CAPEX budget

Official oemof helpers/native features used:
    - oemof.tools.economics.annuity
    - solph.Investment
    - solph.components.GenericStorage
    - invest_relation_input_capacity / invest_relation_output_capacity
    - solph.components.Converter
    - solph.processing.results
================================================================================
"""

import pandas as pd
import pyomo.environ as po
from oemof import solph
from oemof.tools import economics


# ==============================================================================
# STEP 1 - LOAD INPUT DATA
# ==============================================================================

data = pd.read_csv("input_data_office_medium.csv", index_col=0, parse_dates=True)

print("Input loaded")
print("Timesteps:", len(data))
print("Columns  :", list(data.columns))
print()


# ==============================================================================
# STEP 2 - PARAMETERS (Medium building office)
# ==============================================================================

WACC   = 0.04
BUDGET = 300000          # office realistic (~124 EUR/m2).

# PV  [Fraunhofer ISE PV Report 2025: 10-100 kWp rooftop < 1260 EUR/kWp]
CAPEX_PV = 1100           
LIFE_PV  = 25

# ---- PV AREA CONSTRAINT (office) ----
# Office building (~1600 m2 floor). Roof-to-floor ratio for a low-rise
# commercial building ~0.40 (was 0.50 for a house). ~6 m2 module per kWp.
# Only an upper bound; does not force PV investment.
NET_USABLE_AREA_M2 = 1600.0  
PV_AREA_RATIO_M2_PER_M2_LIVING = 0.40 
PV_AREA_PER_KWP_M2 = 6.0
PV_USABLE_AREA_M2  = NET_USABLE_AREA_M2 * PV_AREA_RATIO_M2_PER_M2_LIVING
PV_MAX             = PV_USABLE_AREA_M2 / PV_AREA_PER_KWP_M2

# building peaks (auto from office CSV -> all other caps scale from these)
PEAK_HEAT_KW = float(data["heat_demand_kW"].max())
PEAK_EL_KW   = float(data["el_demand_kW"].max()) 

# Battery  [SolarPower Europe / German C&I 2025]
CAPEX_BAT = 450 
LIFE_BAT  = 15
BAT_EFF   = 0.95
BAT_MIN_SOC = 0.10         # 10 % minimum
BAT_MAX_SOC = 1         # 100% max
BAT_C_RATE  = 1.0

# TES  [Danish Energy Agency, hot-water tank]
CAPEX_TES = 50            # unchanged
LIFE_TES  = 20
TES_EFF   = 0.98
TES_LOSS  = 0.001
TES_C_RATE = 0.25

# Heat pump  [Danish Energy Agency / PyPSA technology-data]
CAPEX_HP = 1200           # unchanged
LIFE_HP  = 20
COP_HP   = 3.0
HP_MAX   = PEAK_HEAT_KW
HP_RAMP_UP_RATE = 0.50
HP_RAMP_DOWN_RATE = 0.50

# CHP  [ASUE BHKW-Kenndaten / Danish Energy Agency, small gas engine]
CAPEX_CHP = 1500          # unchanged
LIFE_CHP  = 20
ETA_EL_CHP = 0.35
ETA_TH_CHP = 0.45
CHP_RAMP_UP_RATE = 0.25
CHP_RAMP_DOWN_RATE = 0.25
CHP_MAX_TH  = PEAK_HEAT_KW
CHP_MAX_GAS = CHP_MAX_TH / ETA_TH_CHP
CHP_MAX_EL  = CHP_MAX_GAS * ETA_EL_CHP        

# AUX gas boiler (gas input, see aux_heater component)
CAPEX_AUX = 150          
LIFE_AUX  = 20
AUX_EFF   = 0.90
AUX_MAX   = PEAK_HEAT_KW

# Storage capacities (Battery, TES): NO upper bound.
# oemof default for Investment(maximum=...) is unbounded. Storage sizing is
# limited only by investment cost and the upfront CAPEX budget.

# Operating costs
GAS_PRICE = 0.07          #[Eurostat non-household DE 2024]
EXPORT_PRICE = data["feedin_tariff_eur_kWh"]
BAT_DISCHARGE_COST = 0.001
TES_DISCHARGE_COST = 0.001

# Aux is kept as a last-resort LP backup by pricing its heat slightly above
# the worst-case grid-powered HP heat cost. This is an economic priority,
# not a binary ON/OFF priority.
HP_WORST_CASE_HEAT_COST = float(data["grid_price_eur_kWh"].max()) / COP_HP
AUX_BASE_HEAT_COST = GAS_PRICE / AUX_EFF
AUX_BACKUP_PENALTY = max(
    0.0,
    HP_WORST_CASE_HEAT_COST - AUX_BASE_HEAT_COST,
) + 0.01
AUX_OUTPUT_COST = AUX_BACKUP_PENALTY

# Explicit excess-heat rejection.
# A very small positive cost makes dumping possible only when useful, while
# avoiding the need to use TES charge/discharge losses as an artificial dump.
HEAT_DUMP_COST = 0.0001


# ==============================================================================
# STEP 3 - OFFICIAL OEMOF ECONOMICS HELPER
# ==============================================================================
# economics.annuity converts overnight CAPEX into annualized investment cost.

ep_pv = economics.annuity(CAPEX_PV, LIFE_PV, WACC)
ep_bat = economics.annuity(CAPEX_BAT, LIFE_BAT, WACC)
ep_tes = economics.annuity(CAPEX_TES, LIFE_TES, WACC)
ep_hp = economics.annuity(CAPEX_HP, LIFE_HP, WACC)
ep_chp = economics.annuity(CAPEX_CHP, LIFE_CHP, WACC)
ep_aux = economics.annuity(CAPEX_AUX, LIFE_AUX, WACC)

# CHP investment reference is gas input, so convert EUR/kW_el/year to EUR/kW_gas/year.
ep_chp_gas = ep_chp * ETA_EL_CHP


print("PV area constraint:")
print(f"  Net usable/living area basis : {NET_USABLE_AREA_M2:.1f} m2")
print(f"  PV area ratio                : {PV_AREA_RATIO_M2_PER_M2_LIVING:.2f} m2_PV/m2_living")
print(f"  Usable PV area               : {PV_USABLE_AREA_M2:.1f} m2")
print(f"  Area per kWp assumption      : {PV_AREA_PER_KWP_M2:.1f} m2/kWp")
print(f"  Derived PV maximum           : {PV_MAX:.2f} kWp")
print()


# ==============================================================================
# STEP 4 - ENERGY SYSTEM + BUSES
# ==============================================================================

es = solph.EnergySystem(timeindex=data.index, infer_last_interval=True)

b_gas = solph.Bus(label="gas")
b_el = solph.Bus(label="electricity")
b_th = solph.Bus(label="heat")

# Dedicated source buses enforce charging-source rules natively:
# - Battery can charge only from PV + CHP electricity.
# - TES can charge only from CHP heat.
b_onsite_el = solph.Bus(label="onsite_electricity")
b_chp_heat = solph.Bus(label="chp_heat")


# ==============================================================================
# STEP 5 - SOURCES
# ==============================================================================

grid_import = solph.components.Source(
    label="grid_import",
    outputs={b_el: solph.Flow(variable_costs=data["grid_price_eur_kWh"])},
)

gas_source = solph.components.Source(
    label="gas_source",
    outputs={b_gas: solph.Flow(variable_costs=GAS_PRICE)},
)

pv = solph.components.Source(
    label="pv",
    outputs={
        b_onsite_el: solph.Flow(
            fix=data["pv_cf"],
            nominal_capacity=solph.Investment(
                ep_costs=ep_pv,
                maximum=PV_MAX,
            ),
        )
    },
)

# Unity-efficiency transfer from onsite generation bus to main electricity bus.
# Because the connection is directed, grid electricity cannot flow backward
# into the onsite bus and therefore cannot charge the battery.
onsite_el_to_main = solph.components.Converter(
    label="onsite_el_to_main",
    inputs={b_onsite_el: solph.Flow()},
    outputs={b_el: solph.Flow()},
    conversion_factors={b_el: 1.0},
)


# ==============================================================================
# STEP 6 - DEMANDS + GRID EXPORT
# ==============================================================================

el_demand = solph.components.Sink(
    label="el_demand",
    inputs={b_el: solph.Flow(fix=data["el_demand_kW"], nominal_capacity=1)},
)

heat_demand = solph.components.Sink(
    label="heat_demand",
    inputs={b_th: solph.Flow(fix=data["heat_demand_kW"], nominal_capacity=1)},
)

grid_export = solph.components.Sink(
    label="grid_export",
    inputs={b_el: solph.Flow(variable_costs=-EXPORT_PRICE)},
)


# Explicit excess-heat sink.
# This is a continuous LP flow. It gives the model a physically transparent
# path for rejecting unavoidable/excess heat, instead of dissipating heat by
# charging and discharging TES at the same time.
heat_dump = solph.components.Sink(
    label="heat_dump",
    inputs={
        b_th: solph.Flow(
            variable_costs=HEAT_DUMP_COST,
        )
    },
)


# ==============================================================================
# STEP 7 - BATTERY (NATIVE LINEAR GENERIC STORAGE)
# ==============================================================================

battery = solph.components.GenericStorage(
    label="battery",
    inputs={
        b_onsite_el: solph.Flow(
            nominal_capacity=solph.Investment(ep_costs=0)
        )
    },
    outputs={
        b_el: solph.Flow(
            nominal_capacity=solph.Investment(ep_costs=0),
            variable_costs=BAT_DISCHARGE_COST,
        )
    },
    nominal_storage_capacity=solph.Investment(
        ep_costs=ep_bat,            # no maximum -> unbounded (oemof default)
    ),
    inflow_conversion_factor=BAT_EFF,
    outflow_conversion_factor=BAT_EFF,
    loss_rate=0.0,
    min_storage_level=BAT_MIN_SOC,
    max_storage_level=BAT_MAX_SOC,
    invest_relation_input_capacity=BAT_C_RATE,
    invest_relation_output_capacity=BAT_C_RATE,
    balanced=True,
)


# ==============================================================================
# STEP 8 - TES (NATIVE LINEAR GENERIC STORAGE)
# ==============================================================================

tes = solph.components.GenericStorage(
    label="tes",
    inputs={
        b_chp_heat: solph.Flow(
            nominal_capacity=solph.Investment(ep_costs=0)
        )
    },
    outputs={
        b_th: solph.Flow(
            nominal_capacity=solph.Investment(ep_costs=0),
            variable_costs=TES_DISCHARGE_COST,
        )
    },
    nominal_storage_capacity=solph.Investment(
        ep_costs=ep_tes,            # no maximum -> unbounded (oemof default)
    ),
    inflow_conversion_factor=TES_EFF,
    outflow_conversion_factor=TES_EFF,
    loss_rate=TES_LOSS,
    invest_relation_input_capacity=TES_C_RATE,
    invest_relation_output_capacity=TES_C_RATE,
    balanced=True,
)


# ==============================================================================
# STEP 9 - HEAT PUMP (CONTINUOUS INVESTMENT + CONTINUOUS DISPATCH)
# ==============================================================================
# No NonConvex object. Therefore no binary commitment variable is created.

heat_pump = solph.components.Converter(
    label="heat_pump",
    inputs={b_el: solph.Flow()},
    outputs={
        b_th: solph.Flow(
            nominal_capacity=solph.Investment(
                ep_costs=ep_hp,
                maximum=HP_MAX,
            ),
            maximum=1.0,
        )
    },
    conversion_factors={b_th: COP_HP},
)


# ==============================================================================
# STEP 10 - CHP (CONTINUOUS INVESTMENT + CONTINUOUS DISPATCH)
# ==============================================================================
# Reference investment flow = gas input. No NonConvex object.

chp = solph.components.Converter(
    label="chp",
    inputs={
        b_gas: solph.Flow(
            nominal_capacity=solph.Investment(
                ep_costs=ep_chp_gas,
                maximum=CHP_MAX_GAS,
            ),
            maximum=1.0,
        )
    },
    outputs={
        b_onsite_el: solph.Flow(),
        b_chp_heat: solph.Flow(),
    },
    conversion_factors={
        b_gas: 1.0,
        b_onsite_el: ETA_EL_CHP,
        b_chp_heat: ETA_TH_CHP,
    },
)

# Unity-efficiency direct route from CHP heat bus to main heat bus.
# CHP heat can either serve the building directly or charge TES.
chp_heat_to_main = solph.components.Converter(
    label="chp_heat_to_main",
    inputs={b_chp_heat: solph.Flow()},
    outputs={b_th: solph.Flow()},
    conversion_factors={b_th: 1.0},
)


# ==============================================================================
# STEP 11 - AUXILIARY HEATER
# ==============================================================================

aux_heater = solph.components.Converter(
    label="aux_heater",
    inputs={b_gas: solph.Flow()},
    outputs={
        b_th: solph.Flow(
            nominal_capacity=solph.Investment(
                ep_costs=ep_aux,
                maximum=AUX_MAX,
            ),
            variable_costs=AUX_OUTPUT_COST,
        )
    },
    conversion_factors={b_th: AUX_EFF},
)


# ==============================================================================
# STEP 12 - ADD COMPONENTS + CREATE MODEL
# ==============================================================================

es.add(
    b_gas, b_el, b_th, b_onsite_el, b_chp_heat,
    grid_import, gas_source, pv,
    onsite_el_to_main, chp_heat_to_main,
    el_demand, heat_demand, grid_export, heat_dump,
    battery, tes, heat_pump, chp, aux_heater,
)

model = solph.Model(es)
print("Linear oemof model created: pure LP, unbounded storage + native charging-source buses.")


# ==============================================================================
# STEP 13 - INVESTMENT VARIABLE REFERENCES
# ==============================================================================

pv_invest = model.InvestmentFlowBlock.invest[pv, b_onsite_el, 0]
bat_invest = model.GenericInvestmentStorageBlock.invest[battery, 0]
tes_invest = model.GenericInvestmentStorageBlock.invest[tes, 0]
hp_invest_th = model.InvestmentFlowBlock.invest[heat_pump, b_th, 0]
chp_invest_gas = model.InvestmentFlowBlock.invest[b_gas, chp, 0]
aux_invest_th = model.InvestmentFlowBlock.invest[aux_heater, b_th, 0]


# ==============================================================================
# STEP 14 - TOTAL UPFRONT CAPEX BUDGET (LINEAR CUSTOM CONSTRAINT)
# ==============================================================================

model.total_upfront_capex_budget = po.Constraint(
    expr=(
        CAPEX_PV * pv_invest
        + CAPEX_BAT * bat_invest
        + CAPEX_TES * tes_invest
        + CAPEX_HP * hp_invest_th
        + CAPEX_CHP * ETA_EL_CHP * chp_invest_gas
        + CAPEX_AUX * aux_invest_th
        <= BUDGET
    )
)


# ==============================================================================
# STEP 15 - LINEAR INVESTMENT-LINKED RAMP LIMITS
# ==============================================================================
# IMPORTANT:
# oemof's native positive_gradient_limit / negative_gradient_limit are built
# in SimpleFlowBlock against f.nominal_capacity. For an investment flow in the
# installed oemof-solph version, f.nominal_capacity is None because the capacity
# is represented by an investment variable in InvestmentFlowBlock.
#
# Therefore, the native gradient helper cannot be used directly together with
# endogenous Investment capacity in this version. These constraints explicitly
# link hourly ramping to the optimized HP/CHP investment capacity.
#
# They remain a PURE LP formulation:
#   flow[t] - flow[t-1] <= constant * investment_variable
#
# There are no binary variables, no NonConvex object and no Big-M terms.
# ==============================================================================

first_t = model.TIMESTEPS.first()


def chp_ramp_up_rule(m, p, t):
    if t == first_t:
        return po.Constraint.Skip
    chp_capacity_el = ETA_EL_CHP * chp_invest_gas
    return (
        m.flow[chp, b_onsite_el, t] - m.flow[chp, b_onsite_el, t - 1]
        <= CHP_RAMP_UP_RATE * chp_capacity_el
    )


def chp_ramp_down_rule(m, p, t):
    if t == first_t:
        return po.Constraint.Skip
    chp_capacity_el = ETA_EL_CHP * chp_invest_gas
    return (
        m.flow[chp, b_onsite_el, t - 1] - m.flow[chp, b_onsite_el, t]
        <= CHP_RAMP_DOWN_RATE * chp_capacity_el
    )


def hp_ramp_up_rule(m, p, t):
    if t == first_t:
        return po.Constraint.Skip
    return (
        m.flow[heat_pump, b_th, t] - m.flow[heat_pump, b_th, t - 1]
        <= HP_RAMP_UP_RATE * hp_invest_th
    )


def hp_ramp_down_rule(m, p, t):
    if t == first_t:
        return po.Constraint.Skip
    return (
        m.flow[heat_pump, b_th, t - 1] - m.flow[heat_pump, b_th, t]
        <= HP_RAMP_DOWN_RATE * hp_invest_th
    )


model.chp_ramp_up = po.Constraint(model.TIMEINDEX, rule=chp_ramp_up_rule)
model.chp_ramp_down = po.Constraint(model.TIMEINDEX, rule=chp_ramp_down_rule)
model.hp_ramp_up = po.Constraint(model.TIMEINDEX, rule=hp_ramp_up_rule)
model.hp_ramp_down = po.Constraint(model.TIMEINDEX, rule=hp_ramp_down_rule)


# ==============================================================================
# STEP 16 - SOLVE PURE LP WITH GUROBI
# ==============================================================================

SOLVER = "gurobi"

print("Solving pure LP with", SOLVER.upper())

model.solve(
    solver=SOLVER,
    solve_kwargs={"tee": True},
)


# ==============================================================================
# STEP 17 - RESULTS
# ==============================================================================

result = solph.processing.results(model)

pv_kw = float(result[(pv, b_onsite_el)]["scalars"]["invest"])
bat_kwh = float(result[(battery, None)]["scalars"]["invest"])
bat_kw = bat_kwh * BAT_C_RATE
tes_kwh = float(result[(tes, None)]["scalars"]["invest"])
hp_kw_th = float(result[(heat_pump, b_th)]["scalars"]["invest"])
chp_kw_gas = float(result[(b_gas, chp)]["scalars"]["invest"])
chp_kw_el = chp_kw_gas * ETA_EL_CHP
aux_kw_th = float(result[(aux_heater, b_th)]["scalars"]["invest"])

pv_capex = CAPEX_PV * pv_kw
bat_capex = CAPEX_BAT * bat_kwh
tes_capex = CAPEX_TES * tes_kwh
hp_capex = CAPEX_HP * hp_kw_th
chp_capex = CAPEX_CHP * chp_kw_el
aux_capex = CAPEX_AUX * aux_kw_th
total_capex = pv_capex + bat_capex + tes_capex + hp_capex + chp_capex + aux_capex

pv_gen = result[(pv, b_onsite_el)]["sequences"]["flow"]
grid_imp = result[(grid_import, b_el)]["sequences"]["flow"]
grid_exp = result[(b_el, grid_export)]["sequences"]["flow"]
onsite_el_direct = result[(onsite_el_to_main, b_el)]["sequences"]["flow"]
chp_heat_direct = result[(chp_heat_to_main, b_th)]["sequences"]["flow"]
heat_dump_flow = result[(b_th, heat_dump)]["sequences"]["flow"]
bat_charge = result[(b_onsite_el, battery)]["sequences"]["flow"]
bat_discharge = result[(battery, b_el)]["sequences"]["flow"]
bat_soc = result[(battery, None)]["sequences"]["storage_content"]
tes_charge = result[(b_chp_heat, tes)]["sequences"]["flow"]
tes_discharge = result[(tes, b_th)]["sequences"]["flow"]
tes_soc = result[(tes, None)]["sequences"]["storage_content"]
hp_el = result[(b_el, heat_pump)]["sequences"]["flow"]
hp_heat = result[(heat_pump, b_th)]["sequences"]["flow"]
chp_gas = result[(b_gas, chp)]["sequences"]["flow"]
chp_el = result[(chp, b_onsite_el)]["sequences"]["flow"]
chp_heat = result[(chp, b_chp_heat)]["sequences"]["flow"]
aux_gas = result[(b_gas, aux_heater)]["sequences"]["flow"]
aux_heat = result[(aux_heater, b_th)]["sequences"]["flow"]

# ==============================================================================
# STEP 18 - VALIDATION + EXCEL EXPORT
# ==============================================================================

out = pd.DataFrame(index=pv_gen.index)
out["PV generation (kW)"] = pv_gen
out["Grid import (kW)"] = grid_imp
out["Grid export (kW)"] = grid_exp
out["Onsite electricity direct to main bus (kW)"] = onsite_el_direct
out["CHP heat direct to main heat bus (kW_th)"] = chp_heat_direct
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
out["El demand (kW)"] = data["el_demand_kW"].reindex(out.index)
out["Heat demand (kW)"] = data["heat_demand_kW"].reindex(out.index)
out["Grid price (EUR/kWh)"] = data["grid_price_eur_kWh"].reindex(out.index)
out["Feed-in tariff (EUR/kWh)"] = data["feedin_tariff_eur_kWh"].reindex(out.index)

# Dedicated-source-bus validations
out["Onsite el source bus balance (kW)"] = (
    pv_gen + chp_el - bat_charge - onsite_el_direct
)

out["CHP heat source bus balance (kW_th)"] = (
    chp_heat - tes_charge - chp_heat_direct
)

# Main building buses
out["El bus balance (kW)"] = (
    grid_imp + onsite_el_direct + bat_discharge
    - out["El demand (kW)"] - hp_el - grid_exp
)

out["Heat bus balance (kW)"] = (
    chp_heat_direct + hp_heat + aux_heat + tes_discharge
    - out["Heat demand (kW)"] - heat_dump_flow
)

FLOW_TOL = 1e-6

# Battery SOC validation (native min_storage_level / max_storage_level)
if bat_kwh > 1e-9:
    bat_soc_frac = bat_soc / bat_kwh
    bat_soc_min_achieved = float(bat_soc_frac.min())
    bat_soc_max_achieved = float(bat_soc_frac.max())
else:
    bat_soc_min_achieved = 0.0
    bat_soc_max_achieved = 0.0
battery_simultaneous_hours = int(
    ((bat_charge > FLOW_TOL) & (bat_discharge > FLOW_TOL)).sum()
)
tes_simultaneous_hours = int(
    ((tes_charge > FLOW_TOL) & (tes_discharge > FLOW_TOL)).sum()
)

print()
print("=" * 70)
print("OEMOF PURE LP - OPTIMAL INVESTMENT")
print("=" * 70)
print(f"PV          : {pv_kw:10.2f} kWp  (area-based max: {PV_MAX:.2f} kWp)")
print(f"Peak heat demand reference : {PEAK_HEAT_KW:.2f} kW_th")
print(f"Peak electric demand       : {PEAK_EL_KW:.2f} kW")
print(f"Battery     : {bat_kwh:10.2f} kWh energy | {bat_kw:10.2f} kW max power")
print(f"TES         : {tes_kwh:10.2f} kWh_th")
print(f"Heat Pump   : {hp_kw_th:10.2f} kW_th")
print(f"CHP         : {chp_kw_el:10.2f} kW_el")
print(f"Aux Heater  : {aux_kw_th:10.2f} kW_th  (fully optimized, minimum = 0)")
print(f"Total CAPEX : {total_capex:10.2f} EUR / {BUDGET:.2f} EUR")
print()
print("Validation")
print("Solver                                    :", SOLVER.upper())
print("Battery simultaneous charge/discharge h :", battery_simultaneous_hours)
print("TES simultaneous charge/discharge h     :", tes_simultaneous_hours)
print("Battery SOC min/max achieved            :",
      f"{bat_soc_min_achieved:.4f} / {bat_soc_max_achieved:.4f}")
print("Annual heat dump (kWh_th)                :", float(heat_dump_flow.sum()))
print("El bus max imbalance (kW)               :", out["El bus balance (kW)"].abs().max())
print("Heat bus max imbalance (kW)             :", out["Heat bus balance (kW)"].abs().max())
print("Onsite el source-bus imbalance (kW)     :", out["Onsite el source bus balance (kW)"].abs().max())
print("CHP heat source-bus imbalance (kW_th)   :", out["CHP heat source bus balance (kW_th)"].abs().max())

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
        None,   # Battery CAPEX is charged on energy capacity, not separately on power.
        tes_capex,
        hp_capex,
        chp_capex,
        aux_capex,
        total_capex,
    ],
})

metrics = pd.DataFrame({
    "Metric": [
        "Model type",
        "Solver",
        "Budget (EUR)",
        "Budget remaining (EUR)",
        "PV net usable/living area basis (m2)",
        "PV area ratio (m2_PV/m2_living)",
        "PV usable area (m2)",
        "PV area per kWp assumption (m2/kWp)",
        "PV area-based maximum capacity (kWp)",
        "PV optimized capacity (kWp)",
        "PV maximum-bound utilization (%)",
        "Peak building heat demand (kW_th)",
        "Peak building electricity demand (kW)",
        "HP maximum capacity bound (kW_th)",
        "CHP maximum thermal bound (kW_th)",
        "CHP maximum electrical bound (kW_el)",
        "Aux maximum capacity bound (kW_th)",
        "Storage capacity upper bound",
        "Aux backup marginal penalty (EUR/kWh_th)",
        "Battery C-rate (1/h)",
        "Battery optimized energy capacity (kWh)",
        "Battery optimized max power (kW)",
        "Battery min SOC setting",
        "Battery max SOC setting",
        "Battery SOC min achieved",
        "Battery SOC max achieved",
        "Aux heater minimum investment (kW_th)",
        "Annual heat dump (kWh_th)",
        "Battery simultaneous charge/discharge hours",
        "TES simultaneous charge/discharge hours",
        "El bus max imbalance (kW)",
        "Heat bus max imbalance (kW)",
        "Onsite electricity source-bus max imbalance (kW)",
        "CHP heat source-bus max imbalance (kW_th)",
    ],
    "Value": [
        "LP",
        SOLVER.upper(),
        BUDGET,
        BUDGET - total_capex,
        NET_USABLE_AREA_M2,
        PV_AREA_RATIO_M2_PER_M2_LIVING,
        PV_USABLE_AREA_M2,
        PV_AREA_PER_KWP_M2,
        PV_MAX,
        pv_kw,
        (pv_kw / PV_MAX * 100.0) if PV_MAX > 0 else 0.0,
        PEAK_HEAT_KW,
        PEAK_EL_KW,
        HP_MAX,
        CHP_MAX_TH,
        CHP_MAX_EL,
        AUX_MAX,
        "None (Battery + TES unbounded, budget only)",
        AUX_BACKUP_PENALTY,
        BAT_C_RATE,
        bat_kwh,
        bat_kw,
        BAT_MIN_SOC,
        BAT_MAX_SOC,
        bat_soc_min_achieved,
        bat_soc_max_achieved,
        0.0,
        float(heat_dump_flow.sum()),
        battery_simultaneous_hours,
        tes_simultaneous_hours,
        float(out["El bus balance (kW)"].abs().max()),
        float(out["Heat bus balance (kW)"].abs().max()),
        float(out["Onsite el source bus balance (kW)"].abs().max()),
        float(out["CHP heat source bus balance (kW_th)"].abs().max()),
    ],
})

out.index.name = "Timestamp"
result_file = "Oemof_LINEAR_LP_EXTENDED_NATIVE_SOURCE_BUSES_No_Storage_Bound.xlsx"
with pd.ExcelWriter(result_file) as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
    metrics.to_excel(writer, sheet_name="Summary", index=False, startrow=len(summary) + 3)
    out.to_excel(writer, sheet_name="Hourly results")

print("Saved:", result_file)
print("Finished.")
