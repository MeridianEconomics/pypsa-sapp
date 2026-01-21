# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
import logging
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytz
import ruamel.yaml
import xarray as xr
from _helpers import (
    BASE_DIR,
    create_dummy_data,
    create_network_topology,
    cycling_shift,
    locate_bus,
    mock_snakemake,
    override_component_attrs,
    prepare_costs,
    safe_divide,
    sanitize_carriers,
    sanitize_locations,
    three_2_two_digits_country,
    two_2_three_digits_country,
)
from prepare_network import add_co2limit

# Environment variables
PYPSAEARTH_DIR = os.environ.get("PYPSAEARTH_DIR")

logger = logging.getLogger(__name__)


def attach_load_from_demand_profiles(n, demand_profiles_path):
    """
    Attach electricity demand to network from demand_profiles.csv.

    Expects CSV with:
      - index = timestamps (parseable to datetime)
      - columns = bus names (e.g. AC bus names)

    Creates Load components (one per bus column) and sets p_set time series.
    """
    demand_df = pd.read_csv(demand_profiles_path, index_col=0, parse_dates=True)

    # Align to network snapshots
    demand_df = demand_df.reindex(pd.to_datetime(n.snapshots))

    if demand_df.isna().any().any():
        # If your profiles are fine, this should not happen; but it's safer to catch early
        missing = demand_df.index[demand_df.isna().any(axis=1)]
        raise ValueError(
            f"demand_profiles has missing values after aligning to network snapshots. "
            f"First missing snapshots: {list(missing[:5])}"
        )

    # Ensure buses exist
    missing_buses = [b for b in demand_df.columns if b not in n.buses.index]
    if missing_buses:
        raise ValueError(
            "demand_profiles columns must match bus names in the network. "
            f"Missing buses (first 10): {missing_buses[:10]}"
        )

    # Remove existing loads on those buses (avoid duplicates / double counting)
    existing = n.loads.index[n.loads.bus.isin(demand_df.columns)]
    if len(existing) > 0:
        n.mremove("Load", existing)

    # Add loads named by bus, bus=bus, and p_set time series
    n.madd("Load", demand_df.columns, bus=demand_df.columns, p_set=demand_df)

    # Optional: mark them as AC loads (useful if later logic expects carrier == "AC")
    n.loads.loc[demand_df.columns, "carrier"] = "AC"


def add_lifetime_wind_solar(n, costs):
    """
    Add lifetime for solar and wind generators.
    """
    for carrier in ["solar", "onwind", "offwind"]:
        gen_i = n.generators.index.str.contains(carrier)
        n.generators.loc[gen_i, "lifetime"] = costs.at[carrier, "lifetime"]



def get(item, investment_year=None):
    """
    Check whether item depends on investment year.
    """
    if isinstance(item, dict):
        return item[investment_year]
    else:
        return item



def average_every_nhours(n, offset):
    # logger.info(f'Resampling the network to {offset}')
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset.casefold()).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset.casefold()).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset.casefold()).max()
                else:
                    pnl[k] = df.resample(offset.casefold()).mean()

    return m



def normalize_by_country(df, droplevel=False):
    """
    Auxiliary function to normalize a dataframe by the country.

    If droplevel is False (default), the country level is added to the
    column index If droplevel is True, the original column format is
    preserved
    """
    ret = df.T.groupby(df.columns.str[:2]).apply(lambda x: x / x.sum().sum()).T
    if droplevel:
        return ret.droplevel(0, axis=1)
    else:
        return ret


def group_by_node(df, multiindex=False):
    """
    Auxiliary function to group a dataframe by the node name.
    """
    ret = df.T.groupby(df.columns.str.split(" ").str[0]).sum().T
    if multiindex:
        ret.columns = pd.MultiIndex.from_tuples(zip(ret.columns.str[:2], ret.columns))
    return ret


def normalize_and_group(df, multiindex=False):
    """
    Function to concatenate normalize_by_country and group_by_node.
    """
    return group_by_node(
        normalize_by_country(df, droplevel=True), multiindex=multiindex
    )


def p_set_from_scaling(col, scaling, energy_totals, nhours):
    """
    Function to create p_set from energy_totals, using the per-unit scaling
    dataframe.
    """
    return 1e6 * scaling.div(nhours, axis=0).mul(energy_totals[col], level=0).droplevel(
        level=0, axis=1
    )



def add_co2_budget(n, co2_budget, investment_year, elec_opts):
    # Check if CO2Limit already exists
    if "CO2Limit" in n.global_constraints.index and co2_budget["override_co2opt"]:
        logger.warning("CO2Limit already exists, value will be overwritten.")
        n.global_constraints.drop(index="CO2Limit", inplace=True)
    else:
        logger.info("CO2Limit already exists, value will not be overwritten.")
        return

    # Get base year emission factor
    factor = (
        co2_budget["year"][investment_year]
        if investment_year in co2_budget["year"]
        else 1.0
    )

    co2base_value = co2_budget["co2base_value"]
    if co2base_value == "co2limit":
        annual_emissions = factor * elec_opts["co2limit"]
    elif co2base_value == "co2base":
        annual_emissions = factor * elec_opts["co2base"]
    elif co2base_value == "absolute":
        annual_emissions = factor
    elif isinstance(co2base_value, float):
        annual_emissions = factor * co2base_value
    else:
        raise ValueError(
            f"co2base_value: {co2base_value} is not an option for co2_budget"
        )

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    logger.info(
        f"Annual emissions for {investment_year} set to {annual_emissions / 1e6:.2f} MtCO₂-eq/year."
    )
    add_co2limit(n, annual_emissions, Nyears)




def get_capacities_from_elec(n, carriers, component):
    """
    Gets capacities and efficiencies for {carrier} in n.{component} that were
    previously assigned in add_electricity.
    """
    component_list = ["generators", "storage_units", "links", "stores"]
    component_dict = {name: getattr(n, name) for name in component_list}
    e_nom_carriers = ["stores"]
    nom_col = {x: "e_nom" if x in e_nom_carriers else "p_nom" for x in component_list}
    eff_col = "efficiency"

    capacity_dict = {}
    efficiency_dict = {}
    node_dict = {}
    for carrier in carriers:
        capacity_dict[carrier] = component_dict[component].query("carrier in @carrier")[
            nom_col[component]
        ]
        efficiency_dict[carrier] = component_dict[component].query(
            "carrier in @carrier"
        )[eff_col]
        node_dict[carrier] = component_dict[component].query("carrier in @carrier")[
            "bus"
        ]

    return capacity_dict, efficiency_dict, node_dict



if __name__ == "__main__":
    if "snakemake" not in globals():
        # from helper import mock_snakemake #TODO remove func from here to helper script
        snakemake = mock_snakemake(
            "prepare_elec_network",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L-4H",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.071,
            demand="AB",
        )

    # Load input network
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    # Fetch the country list from the network
    # countries = list(n.buses.country.unique())
    countries = snakemake.params.countries
    # Locate all the AC buses
    acnodes = n.buses[
        n.buses.carrier == "AC"
    ].index  # TODO if you take nodes from the index of buses of n it's more than pop_layout
    # clustering of regions must be double checked.. refer to regions onshore

    # Add location. TODO: move it into pypsa-earth
    n.buses.location = n.buses.index

    # Set carrier of AC loads
    existing_nodes = [node for node in acnodes if node in n.loads.index]
    if len(existing_nodes) < len(acnodes):
        print(
            f"Warning: For {len(acnodes) - len(existing_nodes)} of {len(acnodes)} nodes there were no load nodes found in network and were skipped."
        )
    n.loads.loc[existing_nodes, "carrier"] = "AC"

    Nyears = n.snapshot_weightings.generators.sum() / 8760

    # Fetch wildcards
    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    demand_sc = snakemake.wildcards.demand  # loading the demand scenario wildcard

    # Prepare the costs dataframe
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.config["costs"],
        snakemake.params.costs["output_currency"],
        snakemake.params.costs["fill_values"],
        Nyears,
        snakemake.params.costs["default_exchange_rate"],
        snakemake.params.costs["future_exchange_rate_strategy"],
        snakemake.params.costs["custom_future_exchange_rate"],
    )


    if snakemake.params.foresight in ["myopic", "perfect"]:
        add_lifetime_wind_solar(n, costs)

    # TODO logging

    energy_totals = pd.read_csv(
        snakemake.input.energy_totals,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )

    ##########################################################################
    ################# Functions adding different carrires ####################
    ##########################################################################


    sopts = snakemake.wildcards.sopts.split("-")

    for o in sopts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break
    
    # Attach electricity-only demand (same convention as add_electricity.py)
    attach_load_from_demand_profiles(n, snakemake.input.demand)


    co2_budget = snakemake.params.co2_budget
    if co2_budget["enable"]:
        add_co2_budget(
            n,
            co2_budget,
            investment_year,
            snakemake.params.electricity,
        )


    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)

    n.export_to_netcdf(snakemake.output[0])

    # TODO changes in case of myopic oversight
