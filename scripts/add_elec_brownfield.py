# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prepares brownfield data from previous planning horizon.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import sanitize_carriers, sanitize_locations
from add_existing_elec_baseyear import add_build_year_to_new_assets
from add_electricity import load_costs
import re
import yaml

# from pypsa.clustering.spatial import normed_or_uniform

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def add_brownfield(n, n_p, year):
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    for c in n_p.iterate_components(["Link", "Generator", "Store", "StorageUnit"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)


def _pypsa_carrier_from_pp(row) -> str:
    fuel = str(row.get("Fueltype", "")).strip().lower()
    tech = str(row.get("Technology", "")).strip().lower()

    if fuel in {"natural gas", "gas"}:
        if "ccgt" in tech or "combined" in tech:
            return "CCGT"
        if "ocgt" in tech or "open" in tech:
            return "OCGT"
        return "gas"

    if fuel in {"hard coal", "lignite", "coal"}:
        return "coal"
    
    if fuel == "oil":
        return "oil"

    if fuel == "hydro":
        if "run" in tech or "run-of-river" in tech or "ror" in tech:
            return "ror"
        if tech == "pumped storage":
            return "phs"
        if tech == "reservoir":
            return "reservoir"
        return "hydro"

    return fuel


def cap_exogenous_generators_and_storage_units(
    n_p,
    n,
    powerplants_csv_path: str,
    ppm_cfg_yaml_path: str,
    year: int,
):
    """
    For horizon `year`, compute surviving plant capacity from powerplants.csv and
    cap/remove EXOGENOUS generators and storage units.

    Endogenous builds are preserved by excluding component names ending with '-YYYY'.
    """
    with open(ppm_cfg_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    fuel_to_life = cfg.get("fuel_to_lifetime", {})

    pp = pd.read_csv(powerplants_csv_path).copy()

    pp["bus"] = pp["bus"].astype(str).str.strip()
    pp["Capacity"] = pd.to_numeric(pp["Capacity"], errors="coerce").fillna(0.0)
    pp["DateIn"] = pd.to_numeric(pp["DateIn"], errors="coerce")
    pp["DateOut"] = pd.to_numeric(pp["DateOut"], errors="coerce")

    # If DateIn is missing OR no lifetime default exists -> treat as "never retires" (inf)
    def _fill_dateout(row):
        di = row["DateIn"]
        do = row["DateOut"]
        if np.isfinite(do):
            return do
        if not np.isfinite(di):
            return np.inf
        fuel = str(row["Fueltype"]).strip()
        life = fuel_to_life.get(fuel, None)
        return (di + float(life)) if life is not None else np.inf

    pp["DateOut_filled"] = pp.apply(_fill_dateout, axis=1)

    pp["carrier"] = pp.apply(_pypsa_carrier_from_pp, axis=1)

    # Surviving plant capacity in this horizon year
    surviving = pp[(pp["DateIn"] <= year) & (pp["DateOut_filled"] > year)]
    surviving_cap = surviving.groupby(["region_id", "carrier"])["Capacity"].sum()

    # Identify exogenous generators in the network (no "-YYYY" suffix)
    idx = n.generators.index.to_series()
    is_endogenous = idx.str.contains(r"-\d{4}$", regex=True)
    exo = n.generators.loc[~is_endogenous].copy()

    removed = 0
    capped = 0

    for g, row in exo.iterrows():
        bus = str(row["bus"])
        bus = bus.replace("_AC", "").replace("_DC", "")
        car = str(row["carrier"]).lower()

        cap = float(surviving_cap.get((bus, car), 0.0))

        if cap <= 0.0:
            n.mremove("Generator", [g])
            if g in n_p.generators.index:
                n_p.mremove("Generator", [g])
            removed += 1
            continue

        old = float(n.generators.at[g, "p_nom"])
        new = min(old, cap)
        if new < old - 1e-6:
            n.generators.at[g, "p_nom"] = new
            n.generators.at[g, "p_nom_min"] = new
            if g in n_p.generators.index:
                n_p.generators.at[g, "p_nom"] = new
                n_p.generators.at[g, "p_nom_min"] = new
                if "p_nom_opt" in n_p.generators.columns and pd.notnull(n_p.generators.at[g, "p_nom_opt"]):
                    n_p.generators.at[g, "p_nom_opt"] = new
            capped += 1

    logger.info(
        f"Exogenous Generator retirement by plant table (year={year}): removed {removed}, capped {capped}"
    )

    # Identify exogenous storage units in the network (no "-YYYY" suffix)
    idx = n.storage_units.index.to_series()
    is_endogenous = idx.str.contains(r"-\d{4}$", regex=True)
    exo = n.storage_units.loc[~is_endogenous].copy()

    removed = 0
    capped = 0

    for su, row in exo.iterrows():
        bus = str(row["bus"])
        bus = bus.replace("_AC", "").replace("_DC", "")
        car = str(row["carrier"]).lower()
        if car == "hydro":
            car = "reservoir"

        cap = float(surviving_cap.get((bus, car), 0.0))

        if cap <= 0.0:
            n.mremove("StorageUnit", [su])
            if su in n_p.storage_units.index:
                n_p.mremove("StorageUnit", [su])
            removed += 1
            continue

        old = float(n.storage_units.at[su, "p_nom"])
        new = min(old, cap)
        if new < old - 1e-6:
            n.storage_units.at[su, "p_nom"] = new
            n.storage_units.at[su, "p_nom_min"] = new
            if su in n_p.storage_units.index:
                n_p.storage_units.at[su, "p_nom"] = new
                n_p.storage_units.at[su, "p_nom_min"] = new
                if "p_nom_opt" in n_p.storage_units.columns and pd.notnull(n_p.storage_units.at[su, "p_nom_opt"]):
                    n_p.storage_units.at[su, "p_nom_opt"] = new
            capped += 1

    logger.info(
        f"Exogenous Storage Unit retirement by plant table (year={year}): removed {removed}, capped {capped}"
    )

def update_costs(n, costs):
    # Generators
    for idx, row in n.generators.iterrows():
        carrier = row.carrier.replace("-ac", "").replace("-dc", "")
        n.generators.at[idx, "marginal_cost"] = costs.at[carrier, "marginal_cost"]

        if row.get("p_nom_extendable", False):
            n.generators.at[idx, "capital_cost"] = costs.at[carrier, "capital_cost"]
        
    # Storage Units
    for idx, row in n.storage_units.iterrows():
        carrier = row.carrier
        n.storage_units.at[idx, "marginal_cost"] = costs.at[carrier, "marginal_cost"]

        if row.get("p_nom_extendable", False):
            n.storage_units.at[idx, "capital_cost"] = costs.at[carrier, "capital_cost"]


def add_myopic_year_emission_prices(n, emission_prices, year):
    emission_price = {'co2':emission_prices["co2_per_year"][year]}
    ep = (
        pd.Series(emission_price).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep

def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep


def disable_grid_expansion_if_limit_hit(n):
    """
    Check if transmission expansion limit is already reached; then turn off.

    In particular, this function checks if the total transmission
    capital cost or volume implied by s_nom_min and p_nom_min are
    numerically close to the respective global limit set in
    n.global_constraints. If so, the nominal capacities are set to the
    minimum and extendable is turned off; the corresponding global
    constraint is then dropped.
    """
    cols = {"cost": "capital_cost", "volume": "length"}
    for limit_type in ["cost", "volume"]:
        glcs = n.global_constraints.query(
            f"type == 'transmission_expansion_{limit_type}_limit'"
        )

        for name, glc in glcs.iterrows():
            total_expansion = (
                (
                    n.lines.query("s_nom_extendable")
                    .eval(f"s_nom_min * {cols[limit_type]}")
                    .sum()
                )
                + (
                    n.links.query("carrier == 'DC' and p_nom_extendable")
                    .eval(f"p_nom_min * {cols[limit_type]}")
                    .sum()
                )
            ).sum()

            # Allow small numerical differences
            if np.abs(glc.constant - total_expansion) / glc.constant < 1e-6:
                logger.info(
                    f"Transmission expansion {limit_type} is already reached, disabling expansion and limit"
                )
                extendable_acs = n.lines.query("s_nom_extendable").index
                n.lines.loc[extendable_acs, "s_nom_extendable"] = False
                n.lines.loc[extendable_acs, "s_nom"] = n.lines.loc[
                    extendable_acs, "s_nom_min"
                ]

                extendable_dcs = n.links.query(
                    "carrier == 'DC' and p_nom_extendable"
                ).index
                n.links.loc[extendable_dcs, "p_nom_extendable"] = False
                n.links.loc[extendable_dcs, "p_nom"] = n.links.loc[
                    extendable_dcs, "p_nom_min"
                ]

                n.global_constraints.drop(name, inplace=True)



if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_elec_brownfield",
            simpl="",
            clusters="10",
            ll="copt",
            opts="Ep-1h",
            planning_horizons="2040",
            discountrate=0.071,
            demand="AB",
        )

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)
    opts = snakemake.wildcards.opts.split("-")

    n = pypsa.Network(snakemake.input.network)
    Nyears = n.snapshot_weightings.generators.sum() / 8760

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p)

    cap_exogenous_generators_and_storage_units(n_p, n, snakemake.input.powerplants, snakemake.input.pm_config, year)

    add_brownfield(n, n_p, year)

    costs = load_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        snakemake.params.electricity,
        Nyears,
    )

    update_costs(n, costs)

    if "co2_per_year" in snakemake.params.costs["emission_prices"]:
        for o in opts:
            if "Ep" in o:
                m = re.findall("[0-9]*\.?[0-9]+$", o)
                logger.info("Setting emission prices according to config value.")
                add_myopic_year_emission_prices(n, snakemake.params.costs["emission_prices"], year)
                break
    else:
        for o in opts:
            if "Ep" in o:
                m = re.findall("[0-9]*\.?[0-9]+$", o)
                if len(m) > 0:
                    logger.info("Setting emission prices according to wildcard value.")
                    add_emission_prices(n, dict(co2=float(m[0])))
                else:
                    logger.info("Setting emission prices according to config value.")
                    add_emission_prices(n, snakemake.params.costs["emission_prices"])
                break

    disable_grid_expansion_if_limit_hit(n)

    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
