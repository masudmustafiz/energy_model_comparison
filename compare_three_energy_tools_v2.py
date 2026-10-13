"""
Compare Calliope, oemof and PyPSA full-year LP outputs.

Outputs (PNG, svg, 600 dpi):
  01_monthly_electric_demand_coverage.png
  02_monthly_heat_demand_coverage.png
  03_combined_operational_event_counts.png
  04a_hourly_HP_ramp_full_year.png
  04b_hourly_CHP_ramp_full_year.png
  06_monthly_grid_and_gas_import_comparison.png
  07_annual_cost_saving_profit.png
  08_input_hourly_electric_heat_demand.png
  annual_summary.csv

Install dependencies:
  pip install pandas numpy matplotlib openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# 1) USER CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "Oemof": BASE_DIR / "Oemof_LINEAR_LP_EXTENDED_NATIVE_SOURCE_BUSES_No_Storage_Bound.xlsx",
    "PyPSA": BASE_DIR / "PyPSA_LINEAR_LP_FINAL_NATIVE_SOC_WINDOW__No_Storage_Bound_results.xlsx",
    "Calliope": BASE_DIR / "Calliope_LP_results.xlsx",
}
INPUT_CSV = BASE_DIR / "input_data_office_medium.csv"
OUTPUT_DIR = BASE_DIR / "comparison_graphs_v6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost assumptions for the pre-model/base case.
# Result data are consistent with gas = 0.07 EUR/kWh_gas.
GAS_PRICE_EUR_PER_KWH = 0.07
BASELINE_GAS_BOILER_EFFICIENCY = 0.90

# Threshold below which a device is treated as OFF / no meaningful ramp.
EPS = 1e-4

# If True, aggregate onsite electricity sent to the main bus is split into PV
# and CHP in proportion to their simultaneous generation. This is an
# attribution method because the exported Excel does not contain separate
# "PV direct to main bus" and "CHP electric direct to main bus" columns.
DECOMPOSE_ONSITE_TO_PV_CHP = True

# =============================================================================
# 2) EXPECTED COLUMN NAMES
# =============================================================================
COL = {
    "timestamp": "Timestamp",
    "pv": "PV generation (kW)",
    "grid_import": "Grid import (kW)",
    "grid_export": "Grid export (kW)",
    "onsite_direct": "Onsite electricity direct to main bus (kW)",
    "chp_heat_direct": "CHP heat direct to main heat bus (kW_th)",
    "battery_charge": "Battery charge (kW)",
    "battery_discharge": "Battery discharge (kW)",
    "tes_charge": "TES charge (kW)",
    "tes_discharge": "TES discharge (kW)",
    "hp_el": "Heat pump electricity (kW)",
    "hp_heat": "Heat pump heat (kW)",
    "chp_gas": "CHP gas input (kW_gas)",
    "chp_el": "CHP electricity (kW)",
    "chp_heat": "CHP heat (kW)",
    "aux_gas": "Aux heater gas (kW_gas)",
    "aux_heat": "Aux heater heat (kW)",
    "el_demand": "El demand (kW)",
    "heat_demand": "Heat demand (kW)",
    "grid_price": "Grid price (EUR/kWh)",
    "fit": "Feed-in tariff (EUR/kWh)",
}

# =============================================================================
# 3) HELPERS
# =============================================================================
def excel_serial_to_datetime(s: pd.Series) -> pd.Series:
    """Robustly convert Excel serial or already-datetime timestamps."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30")
    return pd.to_datetime(s, errors="coerce")


def load_result(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Hourly results", engine="openpyxl")
    missing = [c for c in COL.values() if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns: {missing}")

    df[COL["timestamp"]] = excel_serial_to_datetime(df[COL["timestamp"]])
    df = df.dropna(subset=[COL["timestamp"], COL["el_demand"], COL["heat_demand"]]).copy()
    df = df.sort_values(COL["timestamp"]).reset_index(drop=True)

    for c in df.columns:
        if c != COL["timestamp"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["month"] = df[COL["timestamp"]].dt.to_period("M")
    return df


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["timestamp", "el_demand_kW", "heat_demand_kW", "grid_price_eur_kWh", "feedin_tariff_eur_kWh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def months_as_labels(index) -> list[str]:
    return [pd.Period(x, freq="M").strftime("%b") if not isinstance(x, pd.Period) else x.strftime("%b") for x in index]


def start_stop_flags(power: pd.Series, eps: float = EPS):
    on = power.fillna(0.0).abs() > eps
    prev = on.shift(1, fill_value=False)
    starts = on & ~prev
    stops = ~on & prev
    return starts.astype(int), stops.astype(int)


def ramp_series(power: pd.Series, eps: float = EPS):
    d = power.fillna(0.0).diff().fillna(0.0)
    up = d.where(d > eps, 0.0)
    down = d.where(d < -eps, 0.0)  # negative for plotting
    up_event = (d > eps).astype(int)
    down_event = (d < -eps).astype(int)
    return up, down, up_event, down_event


def monthly_sum(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.groupby("month", observed=True)[columns].sum()


def monthly_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["Battery charge starts"], _ = start_stop_flags(df[COL["battery_charge"]])
    out["Battery discharge starts"], _ = start_stop_flags(df[COL["battery_discharge"]])
    out["TES charge starts"], _ = start_stop_flags(df[COL["tes_charge"]])
    out["TES discharge starts"], _ = start_stop_flags(df[COL["tes_discharge"]])

    out["HP ON"], out["HP OFF"] = start_stop_flags(df[COL["hp_heat"]])
    out["CHP ON"], out["CHP OFF"] = start_stop_flags(df[COL["chp_el"]])
    out["AUX ON"], out["AUX OFF"] = start_stop_flags(df[COL["aux_heat"]])

    _, _, out["HP ramp-up events"], out["HP ramp-down events"] = ramp_series(df[COL["hp_heat"]])
    _, _, out["CHP ramp-up events"], out["CHP ramp-down events"] = ramp_series(df[COL["chp_el"]])

    out["month"] = df["month"].values
    return out.groupby("month", observed=True).sum(numeric_only=True)


def monthly_electric_gross_supply(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly ACTUAL gross electrical production/supply. No demand allocation.

    PV and CHP are their full generated electricity from the Excel file. Grid import
    and battery discharge are shown as additional electrical supply flows. Therefore
    the stacked bar is intentionally NOT forced to equal building demand. Excess can
    go to export, battery charging, HP electricity, etc.
    """
    x = pd.DataFrame({
        "PV generation": df[COL["pv"]].clip(lower=0.0),
        "CHP electricity": df[COL["chp_el"]].clip(lower=0.0),
        "Battery discharge": df[COL["battery_discharge"]].clip(lower=0.0),
        "Grid import": df[COL["grid_import"]].clip(lower=0.0),
        "month": df["month"],
    })
    return x.groupby("month", observed=True).sum(numeric_only=True)


def monthly_heat_gross_supply(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly ACTUAL gross thermal production/supply. No demand allocation.

    CHP heat uses the full `CHP heat` column, not only direct-to-main-bus heat.
    TES discharge is shown as a supply flow. AUX is retained even when very small.
    Thus the stack may be above/below the demand line because storage charging,
    dumping and conversion flows are not hidden.
    """
    x = pd.DataFrame({
        "Heat Pump heat": df[COL["hp_heat"]].clip(lower=0.0),
        "CHP heat": df[COL["chp_heat"]].clip(lower=0.0),
        "TES discharge": df[COL["tes_discharge"]].clip(lower=0.0),
        "AUX heater heat": df[COL["aux_heat"]].clip(lower=0.0),
        "month": df["month"],
    })
    return x.groupby("month", observed=True).sum(numeric_only=True)


def add_common_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, axis="y", alpha=0.25)


# Output quality for Word: a 600-dpi PNG and a vector SVG of every figure.
# The SVG stays sharp at any zoom (Word: Insert > Pictures > select the .svg).
FIG_DPI = 600
SAVE_SVG = True


def save_fig(fig, filename: str):
    path = OUTPUT_DIR / filename
    targets = [path] + ([path.with_suffix(".svg")] if SAVE_SVG else [])
    for target in targets:
        kw = {"bbox_inches": "tight"}
        if target.suffix == ".png":
            kw["dpi"] = FIG_DPI
        try:
            fig.savefig(target, **kw)
        except OSError:
            alt = target.with_name(target.stem + "_new" + target.suffix)
            fig.savefig(alt, **kw)
            print(f"{target.name} is locked (open in another program); saved as {alt.name}")
            target = alt
        print(f"Saved: {target}")
    plt.close(fig)


def global_legend_outside(fig, axes, ncol=1):
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.87, 0.5), frameon=False, ncol=ncol)
    fig.tight_layout(rect=[0.0, 0.0, 0.85, 0.95])


# =============================================================================
# 4) LOAD ALL DATA
# =============================================================================
for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} file: {path}")
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

results = {name: load_result(path) for name, path in FILES.items()}
input_df = load_input(INPUT_CSV)

# =============================================================================
# 5) GRAPH 1: MONTHLY ELECTRIC DEMAND / SUPPLY + GAS USE
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
for ax, (name, df) in zip(axes, results.items()):
    m = monthly_electric_gross_supply(df)
    labels = months_as_labels(m.index)
    x = np.arange(len(m.index))
    m.plot(kind="bar", stacked=True, ax=ax, width=0.78)
    demand_monthly = df.groupby("month", observed=True)[COL["el_demand"]].sum()
    ax.plot(x, demand_monthly.values, color="black", linewidth=2.0, marker="o", markersize=4, label="Electric demand")
    ax.set_title(f"{name} — Electricity", fontweight="bold")
    ax.set_ylabel("Electric energy (kWh)" if ax is axes[0] else "")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, axis="y", alpha=0.22)

    gas_monthly = df.assign(_gas=df[COL["chp_gas"]] + df[COL["aux_gas"]]).groupby("month", observed=True)["_gas"].sum()
    ax2 = ax.twinx()
    ax2.plot(x, gas_monthly.values, color="dimgray", linestyle="--", linewidth=1.8, marker="s", markersize=4, label="Gas import/use")
    ax2.set_ylabel("Gas input (kWh_gas)" if ax is axes[-1] else "")
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

# one common legend above the plotting body
fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
fig.suptitle("Monthly Actual Electricity Generation/Supply vs Demand and Gas Use | Oemof · PyPSA · Calliope", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.84])
save_fig(fig, "01_monthly_electric_demand_coverage.png")

# =============================================================================
# 6) GRAPH 2: MONTHLY HEAT DEMAND / SUPPLY + GAS USE
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
for ax, (name, df) in zip(axes, results.items()):
    m = monthly_heat_gross_supply(df)
    labels = months_as_labels(m.index)
    x = np.arange(len(m.index))
    m.plot(kind="bar", stacked=True, ax=ax, width=0.78)
    demand_monthly = df.groupby("month", observed=True)[COL["heat_demand"]].sum()
    ax.plot(x, demand_monthly.values, color="black", linewidth=2.0, marker="o", markersize=4, label="Heat demand")
    ax.set_title(f"{name} — Heat", fontweight="bold")
    ax.set_ylabel("Thermal energy (kWh_th)" if ax is axes[0] else "")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, axis="y", alpha=0.22)

    gas_monthly = df.assign(_gas=df[COL["chp_gas"]] + df[COL["aux_gas"]]).groupby("month", observed=True)["_gas"].sum()
    ax2 = ax.twinx()
    ax2.plot(x, gas_monthly.values, color="dimgray", linestyle="--", linewidth=1.8, marker="s", markersize=4, label="Gas import/use")
    ax2.set_ylabel("Gas input (kWh_gas)" if ax is axes[-1] else "")
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
fig.suptitle("Monthly Actual Heat Production/Supply vs Demand and Gas Use | Oemof · PyPSA · Calliope", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.84])
save_fig(fig, "02_monthly_heat_demand_coverage.png")

# =============================================================================
# 7) GRAPH 3: COMBINED ANNUAL OPERATIONAL EVENT COUNTS
# =============================================================================
# Starts = number of OFF->ON transitions (one start = one ON/OFF cycle,
# because every start is followed by a stop when no minimum up/down time is set).
annual_events = {}
for name, df in results.items():
    e = monthly_event_counts(df).sum()
    annual_events[name] = {
        "Battery\ncharge starts": int(e["Battery charge starts"]),
        "Battery\ndischarge starts": int(e["Battery discharge starts"]),
        "TES\ncharge starts": int(e["TES charge starts"]),
        "TES\ndischarge starts": int(e["TES discharge starts"]),
        "HP\nstarts": int(e["HP ON"]),
        "HP\nramp-up steps": int(e["HP ramp-up events"]),
        "HP\nramp-down steps": int(e["HP ramp-down events"]),
        "CHP\nstarts": int(e["CHP ON"]),
        "CHP\nramp-up steps": int(e["CHP ramp-up events"]),
        "CHP\nramp-down steps": int(e["CHP ramp-down events"]),
        "AUX Heater\nstarts": int(e["AUX ON"]),
    }

event_df = pd.DataFrame(annual_events)
cats = list(event_df.index)
x = np.arange(len(cats))
width = 0.22
fig, ax = plt.subplots(figsize=(22, 7.5))
for i, name in enumerate(event_df.columns):
    bars = ax.bar(x + (i - 1) * width, event_df[name].values, width=width, label=name)
    ax.bar_label(bars, labels=[f"{int(v):,}" if v else "" for v in event_df[name].values], padding=3, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("Number of events / steps (#)")
ax.set_title("Annual Operational Event Counts | Oemof · PyPSA · Calliope", fontsize=16, fontweight="bold", pad=62)
ax.grid(True, axis="y", alpha=0.20)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)

# visual group separators + labels above graph body
groups = [(0, 1, "Battery"), (2, 3, "TES"), (4, 6, "Heat Pump"), (7, 9, "CHP"), (10, 10, "AUX")]
for gi, (a, b, label) in enumerate(groups):
    center = (a + b) / 2
    ax.text(center, 1.015, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontweight="bold", fontsize=11)
    if gi < len(groups) - 1:
        ax.axvline(b + 0.5, color="0.75", linestyle="--", linewidth=1)
ax.margins(x=0.02)
fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
save_fig(fig, "03_combined_operational_event_counts.png")

# =============================================================================
# 10) GRAPH 4A: HP HOURLY RAMP UP/DOWN, FULL YEAR
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
for ax, (name, df) in zip(axes, results.items()):
    up, down, _, _ = ramp_series(df[COL["hp_heat"]])
    t = df[COL["timestamp"]]
    ax.plot(t, up, linewidth=0.7, label="Ramp up")
    ax.plot(t, down, linewidth=0.7, label="Ramp down")
    ax.axhline(0, linewidth=0.8)
    ax.set_title(name)
    ax.set_ylabel("Δ HP heat output\n(kW_th/h)")
    add_common_time_axis(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.margins(x=0)
fig.suptitle("Hourly Heat Pump ramp-up and ramp-down, full year", fontsize=16)
global_legend_outside(fig, axes)
save_fig(fig, "04a_hourly_HP_ramp_full_year.png")

# =============================================================================
# 11) GRAPH 4B: CHP HOURLY RAMP UP/DOWN, FULL YEAR
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
for ax, (name, df) in zip(axes, results.items()):
    up, down, _, _ = ramp_series(df[COL["chp_el"]])
    t = df[COL["timestamp"]]
    ax.plot(t, up, linewidth=0.7, label="Ramp up")
    ax.plot(t, down, linewidth=0.7, label="Ramp down")
    ax.axhline(0, linewidth=0.8)
    ax.set_title(name)
    ax.set_ylabel("Δ CHP electric output\n(kW_el/h)")
    add_common_time_axis(ax)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.margins(x=0)
fig.suptitle("Hourly CHP ramp-up and ramp-down, full year", fontsize=16)
global_legend_outside(fig, axes)
save_fig(fig, "04b_hourly_CHP_ramp_full_year.png")

# =============================================================================
# 12) GRAPH 5: HOURLY HEAT-DEMAND FULFILMENT IN THE HIGHEST-DEMAND WINTER MONTH
# =============================================================================
# Automatically select the winter month (Dec/Jan/Feb) with the highest total
# heat demand from the original input load profile. This avoids cherry-picking.
winter_input = input_df[input_df["timestamp"].dt.month.isin([12, 1, 2])].copy()
winter_input["period"] = winter_input["timestamp"].dt.to_period("M")
winter_month = winter_input.groupby("period", observed=True)["heat_demand_kW"].sum().idxmax()
winter_label = winter_month.strftime("%B")

fig, axes = plt.subplots(3, 1, figsize=(22, 15), sharex=True, sharey=True)
stack_cols = ["Heat Pump", "CHP heat", "TES discharge", "AUX heater"]

for ax, (name, df) in zip(axes, results.items()):
    sel = df[COL["timestamp"]].dt.to_period("M") == winter_month
    d = df.loc[sel].copy()
    t = d[COL["timestamp"]]

    # Actual hourly thermal outputs/flows from each model Excel file.
    supply = pd.DataFrame({
        "Heat Pump": d[COL["hp_heat"]].clip(lower=0.0).to_numpy(),
        "CHP heat": d[COL["chp_heat"]].clip(lower=0.0).to_numpy(),
        "TES discharge": d[COL["tes_discharge"]].clip(lower=0.0).to_numpy(),
        "AUX heater": d[COL["aux_heat"]].clip(lower=0.0).to_numpy(),
    }, index=t)

    # Stacked hourly production/support. Do NOT force it to equal demand.
    bottom = np.zeros(len(d))
    for c in stack_cols:
        vals = supply[c].to_numpy()
        ax.fill_between(t, bottom, bottom + vals, step="post", alpha=0.78, label=c)
        bottom += vals

    # Heat demand stays independent from production.
    ax.plot(t, d[COL["heat_demand"]], color="black", linewidth=1.7,
            label="Heat demand", zorder=8)

    # TES charging is a heat sink: show below zero so charge/discharge behaviour
    # and TES support are immediately visible without pretending charge is demand.
    tes_charge = d[COL["tes_charge"]].clip(lower=0.0)
    ax.fill_between(t, 0, -tes_charge, step="post", alpha=0.45,
                    label="TES charge (heat stored)")

    # AUX is often tiny. Mark every active AUX hour so it remains visible.
    aux_active = d[COL["aux_heat"]] > EPS
    if aux_active.any():
        ax.scatter(t[aux_active], d.loc[aux_active, COL["heat_demand"]],
                   marker="x", s=28, linewidths=1.2, label="AUX active hour", zorder=10)

    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_title(f"{name} — {winter_label}", fontweight="bold")
    ax.set_ylabel("Thermal power (kW_th)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.margins(x=0)

axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[-1].set_xlabel("Hour / day of selected winter month")

# One clean global legend outside the graph body.
handles, labels = axes[0].get_legend_handles_labels()
# Include AUX-active marker in legend if it appears only in another model.
for ax in axes[1:]:
    h2, l2 = ax.get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h); labels.append(l)
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.875, 0.5), frameon=False)
fig.suptitle(
    f"Hourly Heat Demand Fulfilment and TES Support — Highest-Demand Winter Month: {winter_label}\n"
    "Stack = actual HP + CHP + TES discharge + AUX output; TES charging is shown below zero",
    fontsize=16, fontweight="bold"
)
fig.tight_layout(rect=[0.02, 0.03, 0.86, 0.93])
save_fig(fig, "05_peak_winter_month_hourly_heat_fulfilment.png")

# =============================================================================
# 13) GRAPH 6: HOURLY ELECTRICITY DEMAND FULFILMENT IN THE HIGHEST-DEMAND MONTH
# =============================================================================
# Select the calendar month with the highest total electricity demand from the
# original input load profile. The year is deliberately omitted from plot labels.
input_el_monthly = input_df.assign(period=input_df["timestamp"].dt.to_period("M")).groupby("period", observed=True)["el_demand_kW"].sum()
peak_el_month = input_el_monthly.idxmax()
peak_el_label = peak_el_month.strftime("%B")

fig, axes = plt.subplots(3, 1, figsize=(22, 15), sharex=True, sharey=True)
el_stack_cols = ["PV generation", "CHP electricity", "Grid import", "Battery discharge"]

for ax, (name, df) in zip(axes, results.items()):
    sel = df[COL["timestamp"]].dt.to_period("M") == peak_el_month
    d = df.loc[sel].copy()
    t = d[COL["timestamp"]]

    # Actual hourly electrical production/import/support from each model Excel.
    # These are NOT scaled to demand. Total supply may exceed building demand
    # because electricity can also charge the battery, run the HP, or be exported.
    supply = pd.DataFrame({
        "PV generation": d[COL["pv"]].clip(lower=0.0).to_numpy(),
        "CHP electricity": d[COL["chp_el"]].clip(lower=0.0).to_numpy(),
        "Grid import": d[COL["grid_import"]].clip(lower=0.0).to_numpy(),
        "Battery discharge": d[COL["battery_discharge"]].clip(lower=0.0).to_numpy(),
    }, index=t)

    bottom = np.zeros(len(d))
    for c in el_stack_cols:
        vals = supply[c].to_numpy()
        ax.fill_between(t, bottom, bottom + vals, step="post", alpha=0.78, label=c)
        bottom += vals

    # Building electric demand is independent of gross supply.
    ax.plot(t, d[COL["el_demand"]], color="black", linewidth=1.7,
            label="Electric demand", zorder=8)

    # Battery charging and grid export are electricity sinks/outflows. Showing
    # them below zero explains why gross supply can be above building demand.
    battery_charge = d[COL["battery_charge"]].clip(lower=0.0)
    grid_export = d[COL["grid_export"]].clip(lower=0.0)
    ax.fill_between(t, 0, -battery_charge, step="post", alpha=0.42,
                    label="Battery charge")
    ax.plot(t, -grid_export, linewidth=1.0, alpha=0.8, label="Grid export")

    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_title(f"{name} — {peak_el_label}", fontweight="bold")
    ax.set_ylabel("Electric power (kW)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.margins(x=0)

axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[-1].set_xlabel("Hour / day of selected peak-demand month")

handles, labels = axes[0].get_legend_handles_labels()
for ax in axes[1:]:
    h2, l2 = ax.get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h); labels.append(l)
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.875, 0.5), frameon=False)
fig.suptitle(
    f"Hourly Electricity Demand Fulfilment — Highest-Demand Month: {peak_el_label}\n"
    "Stack = actual PV + CHP electricity + grid import + battery discharge; battery charging is below zero",
    fontsize=16, fontweight="bold"
)
fig.tight_layout(rect=[0.02, 0.03, 0.86, 0.93])
save_fig(fig, "06_peak_electricity_month_hourly_fulfilment.png")

# =============================================================================
# 14) GRAPH 7: MONTHLY GRID ELECTRICITY IMPORT + GAS IMPORT COMPARISON
# =============================================================================
# Actual hourly imports are summed independently for each model.
# Gas import/use = CHP gas input + AUX heater gas input.
months_ref = next(iter(results.values())).groupby("month", observed=True).size().index
month_labels = months_as_labels(months_ref)
x = np.arange(len(months_ref))
width = 0.24

fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)

for i, (name, df) in enumerate(results.items()):
    monthly_grid = df.groupby("month", observed=True)[COL["grid_import"]].sum().reindex(months_ref, fill_value=0.0)
    monthly_gas = (
        df.assign(_gas_import=df[COL["chp_gas"]] + df[COL["aux_gas"]])
          .groupby("month", observed=True)["_gas_import"].sum()
          .reindex(months_ref, fill_value=0.0)
    )
    offset = (i - (len(results)-1)/2) * width
    axes[0].bar(x + offset, monthly_grid.values, width=width, label=name)
    axes[1].bar(x + offset, monthly_gas.values, width=width, label=name)

axes[0].set_title("Monthly grid electricity import", fontweight="bold")
axes[0].set_ylabel("Grid electricity import (kWh)")
axes[0].grid(True, axis="y", alpha=0.25)
axes[1].set_title("Monthly gas import/use", fontweight="bold")
axes[1].set_ylabel("Gas import/use (kWh_gas)")
axes[1].set_xlabel("Month")
axes[1].grid(True, axis="y", alpha=0.25)
axes[1].set_xticks(x)
axes[1].set_xticklabels(month_labels)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
fig.suptitle("Monthly Purchased Energy Comparison | Oemof · PyPSA · Calliope", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.91])
save_fig(fig, "07_monthly_grid_and_gas_import_comparison.png")

# =============================================================================
# 15) GRAPH 8: YEARLY BASELINE COST VS MODEL COST, SAVING AND EXPORT REVENUE
# =============================================================================
# Baseline before optimization:
#   building electricity entirely from grid at hourly grid price
#   building heat entirely from gas boiler
baseline_grid_cost = float((input_df["el_demand_kW"] * input_df["grid_price_eur_kWh"]).sum())
baseline_gas_kwh = float(input_df["heat_demand_kW"].sum() / BASELINE_GAS_BOILER_EFFICIENCY)
baseline_gas_cost = baseline_gas_kwh * GAS_PRICE_EUR_PER_KWH
baseline_total_cost = baseline_grid_cost + baseline_gas_cost

cost_rows = []
for name, df in results.items():
    grid_import_cost = float((df[COL["grid_import"]] * df[COL["grid_price"]]).sum())
    gas_kwh = float((df[COL["chp_gas"]] + df[COL["aux_gas"]]).sum())
    gas_cost = gas_kwh * GAS_PRICE_EUR_PER_KWH
    export_revenue = float((df[COL["grid_export"]] * df[COL["fit"]]).sum())
    purchase_cost = grid_import_cost + gas_cost
    net_model_energy_cost = purchase_cost - export_revenue
    saving = baseline_total_cost - net_model_energy_cost
    saving_pct = 100.0 * saving / baseline_total_cost if baseline_total_cost else np.nan

    cost_rows.append({
        "Tool": name,
        "Baseline grid cost (EUR/y)": baseline_grid_cost,
        "Baseline gas cost (EUR/y)": baseline_gas_cost,
        "Baseline total cost (EUR/y)": baseline_total_cost,
        "Model grid import cost (EUR/y)": grid_import_cost,
        "Model gas cost (EUR/y)": gas_cost,
        "Model purchase cost before export (EUR/y)": purchase_cost,
        "Export revenue (EUR/y)": export_revenue,
        "Net model energy cost (EUR/y)": net_model_energy_cost,
        "Operational saving (EUR/y)": saving,
        "Operational saving (%)": saving_pct,
    })

cost_df = pd.DataFrame(cost_rows).set_index("Tool")

# Annualised investment and remaining objective terms (boiler backup penalty,
# storage discharge cost, heat-dump cost) per tool, taken from the objective
# breakdown of the three models (annual cost table in the report).
ANNUALISED_INVESTMENT = {"Oemof": 13375.83, "PyPSA": 13375.83, "Calliope": 13413.81}
OTHER_MODEL_COSTS = {"Oemof": 28.06 + 14.72 + 0.19, "PyPSA": 28.06 + 14.72 + 0.19, "Calliope": 33.40 + 15.90 + 0.18}
cost_df["Annualised investment (EUR/y)"] = pd.Series(ANNUALISED_INVESTMENT)
cost_df["Other model costs (EUR/y)"] = pd.Series(OTHER_MODEL_COSTS)
cost_df["Total annual cost (EUR/y)"] = (cost_df["Net model energy cost (EUR/y)"]
                                        + cost_df["Annualised investment (EUR/y)"]
                                        + cost_df["Other model costs (EUR/y)"])
cost_df["Net saving (EUR/y)"] = baseline_total_cost - cost_df["Total annual cost (EUR/y)"]
cost_df["Net saving (%)"] = 100.0 * cost_df["Net saving (EUR/y)"] / baseline_total_cost

cases = ["Reference\n(grid + gas boiler)"] + list(cost_df.index)
parts = {
    "Grid electricity": [baseline_grid_cost] + list(cost_df["Model grid import cost (EUR/y)"]),
    "Gas (reference: boiler; optimised: CHP + auxiliary boiler)": [baseline_gas_cost] + list(cost_df["Model gas cost (EUR/y)"]),
    "Annualised investment": [0.0] + list(cost_df["Annualised investment (EUR/y)"]),
    "Other model costs": [0.0] + list(cost_df["Other model costs (EUR/y)"]),
}
export = [0.0] + list(-cost_df["Export revenue (EUR/y)"])
totals = [baseline_total_cost] + list(cost_df["Total annual cost (EUR/y)"])
colors = {"Grid electricity": "#d62728", "Gas (reference: boiler; optimised: CHP + auxiliary boiler)": "#7f7f7f", "Annualised investment": "#1f77b4",
          "Other model costs": "#bcbd22"}

fig, ax = plt.subplots(figsize=(13, 7.5))
x = np.arange(len(cases))
bottom = np.zeros(len(cases))
for name, vals in parts.items():
    vals = np.array(vals)
    ax.bar(x, vals, 0.55, bottom=bottom, label=name, color=colors[name])
    bottom += vals
ax.bar(x, export, 0.55, label="Export revenue (negative)", color="#2ca02c")
ax.scatter(x, totals, marker="D", s=70, color="black", zorder=5, label="Total annual cost")
for i, tot in enumerate(totals):
    ax.annotate(f"Total {tot:,.0f}", (x[i], tot), xytext=(14, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    if i > 0:
        ns = baseline_total_cost - tot
        ax.annotate(f"Net saving\n{ns:,.0f} EUR/a ({100*ns/baseline_total_cost:.0f} %)",
                    (x[i], bottom[i]), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#1f5f1f")
ax.axhline(0, color="0.3", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cases)
ax.set_ylabel("EUR per year")
ax.set_ylim(min(export) * 1.3, baseline_total_cost * 1.15)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
ax.set_title("Annual cost of the reference and the optimised system (investment included)\n"
             f"Grid electricity at the hourly price; gas {GAS_PRICE_EUR_PER_KWH:.3f} EUR/kWh; "
             f"reference boiler efficiency {BASELINE_GAS_BOILER_EFFICIENCY:.0%}", fontsize=13)
fig.tight_layout()
save_fig(fig, "08_annual_cost_saving_profit.png")

# =============================================================================
# 16) GRAPH 9: INPUT DATA - FULL-YEAR HOURLY ELECTRIC + HEAT DEMAND
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
axes[0].plot(input_df["timestamp"], input_df["el_demand_kW"], linewidth=0.75, label="Electric demand")
axes[0].set_ylabel("kW")
axes[0].set_title("Hourly electric demand")
axes[0].grid(True, axis="y", alpha=0.25)
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

axes[1].plot(input_df["timestamp"], input_df["heat_demand_kW"], linewidth=0.75, label="Heat demand")
axes[1].set_ylabel("kW_th")
axes[1].set_title("Hourly heat demand")
axes[1].grid(True, axis="y", alpha=0.25)
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[1].set_xlabel("Month")
axes[1].set_xlim(input_df["timestamp"].iloc[0], input_df["timestamp"].iloc[-1])
axes[1].margins(x=0)

fig.suptitle("Input load profile: full-year hourly electricity and heat demand", fontsize=16)
fig.tight_layout(rect=[0, 0, 0.88, 0.95])
save_fig(fig, "09_input_hourly_electric_heat_demand.png")

# =============================================================================
# 17) CSV SUMMARY: COSTS + YEARLY EVENT TOTALS
# =============================================================================
summary_rows = []
for name, df in results.items():
    events = monthly_event_counts(df).sum()
    r = cost_df.loc[name].to_dict()
    r.update({
        "Tool": name,
        "Annual electric demand (kWh)": float(df[COL["el_demand"]].sum()),
        "Annual heat demand (kWh_th)": float(df[COL["heat_demand"]].sum()),
        "Battery charge starts": int(events["Battery charge starts"]),
        "Battery discharge starts": int(events["Battery discharge starts"]),
        "TES charge starts": int(events["TES charge starts"]),
        "TES discharge starts": int(events["TES discharge starts"]),
        "HP ON": int(events["HP ON"]),
        "HP OFF": int(events["HP OFF"]),
        "HP ramp-up events": int(events["HP ramp-up events"]),
        "HP ramp-down events": int(events["HP ramp-down events"]),
        "CHP ON": int(events["CHP ON"]),
        "CHP OFF": int(events["CHP OFF"]),
        "CHP ramp-up events": int(events["CHP ramp-up events"]),
        "CHP ramp-down events": int(events["CHP ramp-down events"]),
        "AUX ON": int(events["AUX ON"]),
        "AUX OFF": int(events["AUX OFF"]),
    })
    summary_rows.append(r)

summary_df = pd.DataFrame(summary_rows).set_index("Tool")
summary_path = OUTPUT_DIR / "annual_summary.csv"
summary_df.to_csv(summary_path)
print(f"Saved: {summary_path}")

print("\nDone. All graphs and the summary CSV are in:")
print(OUTPUT_DIR)
print("\nAnnual cost summary:")
print(cost_df.round(2).to_string())