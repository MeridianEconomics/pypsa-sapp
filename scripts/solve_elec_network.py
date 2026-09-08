# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Solves linear optimal power flow for a network iteratively while updating
reactances.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: /img/results.png
        :width: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning)
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`prepare_network` and :mod:`solve_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances on values of corresponding flows.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors introduced through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
"""
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging, create_logger
from linopy import merge
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.abstract import optimize_transmission_expansion_iteratively
from pypsa.optimization.optimize import optimize
from add_electricity import load_costs, update_transmission_costs

logger = create_logger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

PYPSAEARTH_DIR = os.environ.get("PYPSAEARTH_DIR")


def get_load_shedding_capacity(n, safety_margin=1.2):
    """
    Calculate required load shedding p_nom per bus based on the
    maximum aggregated load observed in any snapshot.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    safety_margin : float, default 1.2
        Safety factor to apply to the maximum load

    Returns
    -------
    pd.Series
        Required p_nom per bus for load shedding.
    """

    load_shedding_p_nom = pd.Series(0.0, index=n.buses.index)

    for bus_name, bus_loads in n.loads.groupby("bus"):

        if not n.loads_t.p_set.empty:
            bus_load_timeseries = n.loads_t.p_set[
                bus_loads.index.intersection(n.loads_t.p_set.columns)
            ]
            # Sum loads across all components at this bus for each snapshot
            total_load_per_snapshot = bus_load_timeseries.sum(axis=1)
            max_total_load = total_load_per_snapshot.max()
        else:
            max_total_load = bus_loads["p_set"].sum()

        required_p_nom = max_total_load * safety_margin

        load_shedding_p_nom[bus_name] = required_p_nom

    return load_shedding_p_nom


def prepare_network(n, opts, solve_opts, config):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if "lv_limit" in n.global_constraints.index:
        n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
        n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

    if solve_opts.get("load_shedding"):
        required_p_nom = get_load_shedding_capacity(n, safety_margin=1.2)
        n.add("Carrier", "load shedding", color="#dd2e23", nice_name="Load shedding")
        n.madd(
            "Generator",
            n.buses.index,
            " load shedding",
            bus=n.buses.index,
            carrier="load shedding",
            sign=1,
            marginal_cost=solve_opts.get("load_shedding") * 1000,  # convert to Eur/MWh
            p_nom=required_p_nom.reindex(n.buses.index, fill_value=0.5e6),
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if snakemake.config["foresight"] == "myopic":
        add_land_use_constraint(n)

    if "lim" in snakemake.wildcards.ll[1:]:  # defined allows capacity expansion and later under solve_elec_network.py enforce s_nom min/max constraints
        add_line_limit_constraints(n, snakemake.wildcards.ll[1:], snakemake.config)


    return n


def force_transfer_model_only(n):
    n.model.remove_constraints("Kirchhoff-Voltage-Law")

def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.
    Parameter include_existing in config.yaml decides whether existing capacities
    are considered in the CCL constraints. Default is false.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [CCL-Co2L-24H]
    electricity:
        agg_p_nom_limits:
            file: data/agg_p_nom_minmax.csv
            include_existing: false
    """
    agg_p_nom_limits = config["electricity"].get("agg_p_nom_limits")

    try:
        agg_p_nom_minmax = pd.read_csv(
            snakemake.input.agg_p_nom_minmax, index_col=list(range(2)), header=[0, 1]
        )[snakemake.wildcards.planning_horizons]
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity limits per country in "
            "config['electricity']['agg_p_nom_limit']."
        )
    logger.info(
        "Adding per carrier generation capacity constraints for " "individual countries"
    )

    capacity_variable = n.model["Generator-p_nom"]

    # get carriers to which CCL constraints apply
    ccl_carriers = agg_p_nom_minmax.index.get_level_values(1).unique()
    ext_carriers = n.generators.query("p_nom_extendable").carrier.unique()
    ccl_carriers = ccl_carriers[ccl_carriers.isin(ext_carriers)]

    # If no CCL carriers found, return early
    if not ccl_carriers.any():
        logger.info(
            "No CCL carriers found that are extendable. Skipping CCL constraints."
        )
        return

    # Get extendable generators for relevant carriers
    gens = n.generators[n.generators.carrier.isin(ccl_carriers)]
    gens = gens.rename_axis(index="Generator-ext")

    # Prepare country and carrier grouper
    grouper = pd.concat(
        [gens.bus.map(n.buses.country).rename("country"), gens.carrier], axis=1
    )

    # Prepare LHS
    lhs = capacity_variable.groupby(grouper).sum()

    # Obtain existing capacities
    existing_capacities = gens.p_nom.groupby(
        [grouper["country"], grouper["carrier"]]
    ).sum()

    # Obtain minimum and maximum constraint limits
    min_values = agg_p_nom_minmax["min"]
    max_values = agg_p_nom_minmax["max"]

    # Adjust limits if existing capacities are considered
    if agg_p_nom_limits.get("include_existing", False):
        min_values = (min_values - existing_capacities).clip(lower=0)
        max_values = (max_values - existing_capacities).clip(lower=0)
        logger.info(
            f"Considered existing capacities in CCL constraints for carrier {c}."
        )

    # Convert limits to xarray for masking
    min_values = xr.DataArray(min_values.dropna()).rename(dim_0="group")
    max_values = xr.DataArray(max_values.dropna()).rename(dim_0="group")

    # Valid constraints
    valid_min_index = min_values.indexes["group"].intersection(lhs.indexes["group"])
    valid_max_index = max_values.indexes["group"].intersection(lhs.indexes["group"])

    if not valid_min_index.empty:
        n.model.add_constraints(
            lhs.sel(group=valid_min_index) >= min_values.loc[valid_min_index],
            name="agg_p_nom_min",
        )

    if not valid_max_index.empty:
        n.model.add_constraints(
            lhs.sel(group=valid_max_index) <= max_values.loc[valid_max_index],
            name="agg_p_nom_max",
        )


def apply_single_line_derating(n, name, SIL, stability_limit):
    # Multiply the SIL with the St Clair curve to get the line limt as a function of distance
    length = n.lines.loc[name, "length"]# in km
    
    if stability_limit == 'SIL':
        return  SIL / n.lines.loc[name, "s_nom"]
    elif stability_limit == 'SC':
        st_clair = np.minimum(3 * SIL, SIL * 53.736 * (length ** -0.65)) # digitised from https://www.researchgate.net/figure/The-St-Clair-curve-as-based-on-the-results-of-14-retrieved-from-15-is-used-to_fig3_318692193
        return st_clair / n.lines.loc[name, "s_nom"]

   
from pypsa.geo import haversine_pts

def add_custom_lines(n, b0, b1, limit, costs, lines_config, base_voltage):

    length_factor = lines_config["length_factor"]
    linetype = lines_config["ac_types"][base_voltage]
    hvac_cost = costs.at["HVAC overhead", "capital_cost"]

    assert b0 in n.buses.index and b1 in n.buses.index, (b0, b1)
    length = length_factor * haversine_pts(
        n.buses.loc[b0, ["x", "y"]].values,
        n.buses.loc[b1, ["x", "y"]].values,
    )

    name = b0 + '-' + b1
    n.add(
        "Line", name,
        bus0=b0, bus1=b1,
        type=linetype,
        length=length,
        carrier="AC",
        s_max_pu=lines_config["s_max_pu"],
        s_nom_extendable=True,
        s_nom_min=limit["min"],
        s_nom_max=limit["max"],
        capital_cost=length * hvac_cost,
    )

    n.lines.loc[name, "v_nom"] = base_voltage
    n.lines.loc[name, "i_nom"] = n.line_types.i_nom[linetype]
    n.lines.loc[name, "underwater_fraction"] = 0.0

    n.lines.loc[name, "s_nom"] = (
            np.sqrt(3)
            * n.line_types.i_nom[n.lines.loc[name, "type"]]
            * (n.lines.loc[name, "v_nom"] 
            * n.lines.loc[name, "num_parallel"])
        )

    stability_limit=None
    if lines_config["limits"] == "SIL":
        stability_limit = "SIL"
    elif lines_config["limits"] == "St Clair":
        stability_limit = "SC"

    if stability_limit is not None:
        # required for scaling SIL to single voltage level in simplify_network.py
        x_per_length = n.line_types.x_per_length[n.lines.loc[name, "type"]]
        c_per_length = n.line_types.c_per_length[n.lines.loc[name, "type"]]
        b_per_length = (
            2
            * np.pi
            * lines_config["default_frequency"]
            * c_per_length
            * 1e-9
        )

        SIL = n.lines.loc[name, "v_nom"]**2 / np.sqrt(x_per_length / b_per_length) * n.lines.loc[name, "num_parallel"]
        stability_derating = apply_single_line_derating(n, name, SIL, stability_limit)
        # stability_derating = stability_derating

        n.lines.loc[name, "capital_cost"] = n.lines.loc[name, "capital_cost"] / stability_derating

        logger.info(f"Applied additional stability limit to lines according to {stability_limit.replace('SIL','Surge Impedance Loading').replace('SC','St Clair')} method.")


def add_line_limit_constraints(n, factor, config):
    """

    Add minimum and maximum line nominal capacity between buses levels. Opts and path for agg_s_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_s_nom_minmax.csv. The bus names are based on the clustered network and the limits are applied to the sum of line capacities between two buses.
    must be specified by planning horizon in the csv file.


    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        ll: [clim-CP] - reads sub scenario CP (Copper Plate) and applies limits
    electricity:
        agg_s_nom_limits:
            file: data/agg_s_nom_minmax.csv
            include_existing: false
            fix_lines: false
    """
    agg_s_nom_limits = config["electricity"].get("agg_s_nom_limits")
    scenario = factor.split("-")[1]

    try:
        agg_s_nom_minmax = pd.read_csv(
            snakemake.input.agg_s_nom_minmax, index_col=list(range(3)), header=[0, 1]
        ).loc[scenario, snakemake.wildcards.planning_horizons]
        agg_s_nom_minmax = agg_s_nom_minmax.apply(pd.to_numeric, errors="coerce")
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity line limits in"
            "config['electricity']['agg_s_nom_limits']."
        )
    logger.info(
        "Adding custom specified line limits between clustered buses."
    )
    lines_added = 0

    for (bus0, bus1), limits in agg_s_nom_minmax.iterrows():
        mask = ((n.lines.bus0 == bus0) & (n.lines.bus1 == bus1)) | \
               ((n.lines.bus0 == bus1) & (n.lines.bus1 == bus0))
        if mask.any():
            n.lines.loc[mask, "s_nom_min"] = limits["min"]
            n.lines.loc[mask, "s_nom_max"] = limits["max"]
        else:
            add_custom_lines(n, bus0, bus1, limits, costs, snakemake.params.lines, snakemake.params.electricity["base_voltage"])
            lines_added += 1

    if lines_added > 0:
        logger.info(
                "Custom line limits were specified for missing lines"
                f"Missing lines added: {lines_added}"
            )

    factor = n.lines.s_nom_min / n.lines.s_nom # compare forced capacity to the default and update num_parallel lines
    n.lines.num_parallel = factor * n.lines.num_parallel

    if config["electricity"]["agg_s_nom_limits"]["fix_lines"]:
        for (bus0, bus1), limits in agg_s_nom_minmax.iterrows():
            mask = ((n.lines.bus0 == bus0) & (n.lines.bus1 == bus1)) | \
                ((n.lines.bus0 == bus1) & (n.lines.bus1 == bus0))
            n.lines.loc[mask, "s_nom"] = limits["min"]

        n.lines.s_nom_extendable = False

    if config["electricity"]["agg_s_nom_limits"]["remove_external_lines"]:
        valid_pairs = set(agg_s_nom_minmax.index) | {(b, a) for a, b in agg_s_nom_minmax.index}

        # remove any line whose (bus0, bus1) is not a valid pair
        in_agg = n.lines.apply(lambda l: (l.bus0, l.bus1) in valid_pairs, axis=1)
        to_remove = n.lines.index[~in_agg]
        n.mremove("Line", to_remove)

        logger.info(
                        "Lines not in agg_s_nom_minmax were dropped"
                        f"Lines dropped: {len(to_remove)}"
                    )




def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    dispatch_variable = n.model["Generator-p"]
    lhs_gen = (
        (dispatch_variable * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # the current formulation implies that the available hydro power is (inflow - spillage)
    # it implies efficiency_dispatch is 1 which is not quite general
    # see https://github.com/pypsa-meets-earth/pypsa-earth/issues/1245 for possible improvements
    if not n.storage_units_t.inflow.empty:
        spillage_variable = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage_variable * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    capacity_variable = n.model["Generator-p_nom"]
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin_constraint(n, sns, config):
    """
    Build reserve margin constraints based on the formulation
    as suggested in GenX
    https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf
    It implies that the reserve margin also accounts for optimal
    dispatch of distributed energy resources (DERs) and demand response
    which is a novel feature of GenX.
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (
            p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
        ).sum("Generator")

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")


def update_capacity_constraint(n):
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve

    # TODO check if `p_max_pu[ext_i]` is safe for empty `ext_i` and drop if cause in case
    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].rename(
            {"Generator-ext": "Generator"}
        )
        lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="gen_updated_capacity_constraint")


def add_operational_reserve_margin(n, sns, config):
    """
    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """

    add_operational_reserve_margin_constraint(n, sns, config)

    update_capacity_constraint(n)


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_RES_constraints(n, res_share, config):
    """
    The constraint ensures that a predefined share of power is generated
    by renewable sources

    Parameters
    ----------
        n : pypsa.Network
        res_share: float
        config : dict
    """

    logger.warning(
        "The add_RES_constraints() is still work in progress. "
        "Unexpected results might be incurred, particularly if "
        "temporal clustering is applied or if an unexpected change of technologies "
        "is subject to future improvements."
    )

    renew_techs = config["electricity"]["renewable_carriers"]

    charger = ["battery charger"]
    discharger = ["battery discharger"]

    ren_gen = n.generators.query("carrier in @renew_techs")
    ren_stores = n.storage_units.query("carrier in @renew_techs")
    ren_charger = n.links.query("carrier in @charger")
    ren_discharger = n.links.query("carrier in @discharger")

    gens_i = ren_gen.index
    stores_i = ren_stores.index
    charger_i = ren_charger.index
    discharger_i = ren_discharger.index

    stores_t_weights = n.snapshot_weightings.stores

    lgrouper = n.loads.bus.map(n.buses.country)
    ggrouper = ren_gen.bus.map(n.buses.country)
    sgrouper = ren_stores.bus.map(n.buses.country)
    cgrouper = ren_charger.bus0.map(n.buses.country)
    dgrouper = ren_discharger.bus0.map(n.buses.country)

    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    rhs = res_share * load

    # Generators
    lhs_gen = (
        (n.model["Generator-p"].loc[:, gens_i] * n.snapshot_weightings.generators)
        .groupby(ggrouper.to_xarray())
        .sum()
    )

    # StorageUnits
    store_disp_expr = (
        n.model["StorageUnit-p_dispatch"].loc[:, stores_i] * stores_t_weights
    )
    store_expr = n.model["StorageUnit-p_store"].loc[:, stores_i] * stores_t_weights
    charge_expr = n.model["Link-p"].loc[:, charger_i] * stores_t_weights.apply(
        lambda r: r * n.links.loc[charger_i].efficiency
    )
    discharge_expr = n.model["Link-p"].loc[:, discharger_i] * stores_t_weights.apply(
        lambda r: r * n.links.loc[discharger_i].efficiency
    )

    lhs_dispatch = store_disp_expr.groupby(sgrouper).sum()
    lhs_store = store_expr.groupby(sgrouper).sum()

    # Stores (or their resp. Link components)
    # Note that the variables "p0" and "p1" currently do not exist.
    # Thus, p0 and p1 must be derived from "p" (which exists), taking into account the link efficiency.
    lhs_charge = charge_expr.groupby(cgrouper).sum()

    lhs_discharge = discharge_expr.groupby(cgrouper).sum()

    lhs = lhs_gen + lhs_dispatch - lhs_store - lhs_charge + lhs_discharge

    n.model.add_constraints(lhs == rhs, name="res_share")


def add_land_use_constraint(n):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)
    
    g = n.generators

    mask = g.p_nom_extendable & g.p_nom_max.notna()

    # Ensure upper bound is never below what's already installed / required
    g.loc[mask, "p_nom_max"] = (
        g.loc[mask, ["p_nom_max", "p_nom"]]
        .max(axis=1)
    )

    if "p_nom_min" in g.columns:
        g.loc[mask, "p_nom_max"] = (
            g.loc[mask, ["p_nom_max", "p_nom_min"]]
            .max(axis=1)
        )


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "solar rooftop", "onwind", "offwind-ac", "offwind-dc"]:
        existing = (
            n.generators.loc[n.generators.carrier == carrier, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "solar rooftop", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def add_existing(n):
    if snakemake.wildcards["planning_horizons"] == "2050":
        directory = (
            "results/"
            + "Existing_capacities/"
            + snakemake.config["run"].replace("2050", "2030")
        )
        n_name = (
            snakemake.input.network.split("/")[-1]
            .replace(str(snakemake.config["scenario"]["clusters"][0]), "")
            .replace(str(snakemake.config["costs"]["discountrate"][0]), "")
            .replace("_presec", "")
            .replace(".nc", ".csv")
        )

        # n_name = snakemake.input.network.split("/")[-1].replace(str(snakemake.config["scenario"]["clusters"][0]), "").\
        #     replace(".nc", ".csv").replace(str(snakemake.config["costs"]["discountrate"][0]), "")
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)

        for tech in snakemake.config["custom_data"]["renewables"]:
            # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
            existing_res = df.loc[tech]
            existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
            tech_index = n.generators[n.generators.carrier == tech].index
            n.generators.loc[tech_index, tech] = existing_res


def add_lossy_bidirectional_link_constraints(n: pypsa.components.Network) -> None:
    """
    Ensures that the two links simulating a bidirectional_link are extended the same amount.
    """

    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    # ensure that the 'reversed' column is boolean and identify all link carriers that have 'reversed' links
    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    # get the indices of all forward links (non-reversed), that have a reversed counterpart
    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    # function to get backward (reversed) indices corresponding to forward links
    # this function is required to properly interact with the myopic naming scheme
    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    # get the indices of all backward links (reversed)
    backward_i = get_backward_i(forward_i)

    # get the p_nom optimization variables for the links using the get_var function
    links_p_nom = n.model["Link-p_nom"]

    # only consider forward and backward links that are present in the optimization variables
    subset_forward = forward_i.intersection(links_p_nom.indexes["Link-ext"])
    subset_backward = backward_i.intersection(links_p_nom.indexes["Link-ext"])

    # ensure we have a matching number of forward and backward links
    if len(subset_forward) != len(subset_backward):
        raise ValueError("Mismatch between forward and backward links.")

    # define the lefthand side of the constrain p_nom (forward) - p_nom (backward) = 0
    # this ensures that the forward links always have the same maximum nominal power as their backward counterpart
    lhs = links_p_nom.loc[backward_i] - links_p_nom.loc[forward_i]

    # add the constraint to the PySPA model
    n.model.add_constraints(lhs == 0, name="Link-bidirectional_sync")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.linopf.network_lopf``.

    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "RES" in o:
            res_share = float(re.findall(r"[0-9]*\.?[0-9]+$", o)[0])
            add_RES_constraints(n, res_share, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)

    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)

    if config["solving"]["options"]["formulation"] =="transport":
        force_transfer_model_only(n)

        
def add_ramp_rates(n, tech, ramp_limit_up, ramp_limit_down):
    i = n.generators.index[n.generators.carrier == tech]

    n.generators.loc[i, "ramp_limit_up"] = ramp_limit_up
    n.generators.loc[i, "ramp_limit_down"] = ramp_limit_down

    logger.info(
        f"Applied ramp limits to {len(i)} {i} generators: "
        f"up={ramp_limit_up}, down={ramp_limit_down} per snapshot."
    )


def solve_network(n, config, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if skip_iterations or cf_solving.get("formulation", {}) == "transport":
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = cf_solving.get("track_iterations", False)
        kwargs["min_iterations"] = cf_solving.get("min_iterations", 4)
        kwargs["max_iterations"] = cf_solving.get("max_iterations", 6)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if status != "ok":  # and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_elec_network_myopic",
            simpl="",
            clusters="10",
            ll="clim-SAPPFIXED",
            opts="CCL-Ep-1h-EQ0.5c",
            planning_horizons="2030",
            discountrate="0.096",
            demand="AB",
            configfile="config.yaml",
        )

    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]
    costs = load_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        snakemake.params.electricity,
    )

    n = pypsa.Network(snakemake.input.network)

    if "ramp_limits" in snakemake.params.electricity:
        for tech in snakemake.params.electricity["ramp_limits"].keys():
            ramp_limits = snakemake.params.electricity["ramp_limits"][tech]
            ramp_limit_up = ramp_limits.get("ramp_limit_up", 0.1)
            ramp_limit_down = ramp_limits.get("ramp_limit_down", 0.1)

            add_ramp_rates(n, tech, ramp_limit_up, ramp_limit_down)


    if snakemake.params.augmented_line_connection.get("add_to_snakefile"):
        if not n.lines.empty:
            n.lines.loc[n.lines.index.str.contains("new"), "s_nom_min"] = (
                snakemake.params.augmented_line_connection.get("min_expansion")
            )

    if (
        snakemake.config["custom_data"]["add_existing"]
        and snakemake.wildcards.planning_horizons == "2050"
    ):
        add_existing(n)



    n = prepare_network(n, opts, solve_opts, config=solve_opts)

    mask = (n.links.bus0 == "ZA.Gauteng_AC") & (n.links.bus1 == "MZ._AC")
    n.mremove("Link", n.links.index[mask])

    mask = (n.links.bus1 == "ZA.Gauteng_AC") & (n.links.bus0 == "MZ._AC")
    n.mremove("Link", n.links.index[mask])

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        log_fn=snakemake.log.solver,
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Objective function: {n.objective}")
    logger.info(f"Objective constant: {n.objective_constant}")
