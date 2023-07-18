import time
import os
import pandas as pd
from energy_network import EnergyNetwork

if __name__ == '__main__':
    """
    This script initializes the simulation, imports the network data, and plots the initial network.
    """

    # pd.set_option("display.max_rows", None, "display.max_columns", None)  # activate to print every row of a dataframe

    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.")

    # Initialization of the simulation
    year = 2050
    snapshots = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="H")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'Data')

    h2_scenario = 'train+buses'  # ['stock', 'buses', 'stock+buses', 'train', 'train+buses', 'stock+buses+train', 'None']
    h2_installations = None
    h2_bus_scenario = None
    nb_station = None
    nb_disp = None
    stations = {}
    if "bus" in h2_scenario:
        h2_bus_scenario = "freqA"  # freqA, freqB
        nb_station = 2  # 2 ou 3
        nb_disp = 3  # 1, 2 ou 3 par station
    extension_production = False  # if True, the capacity of some generators is extendable

    # Import of the network
    tic = time.time()
    network = EnergyNetwork(snapshots)
    sector_base, sector_new = network.import_network(data_dir, h2=h2_scenario, h2bus=h2_bus_scenario,
                                                     h2station=nb_station, h2disp=nb_disp, h2size=h2_installations,
                                                     ext=extension_production)
    toc = time.time()
    print("INFO: Importing data took {} seconds.".format(toc - tic))

    # raise ValueError('ERROR: STOP.')

    network.plot_network('initial', False, False, False)

    # Optimization of the system
    obj = 'cost'  # cost, env, multi
    limit_water = None

    solver_options = {'Method': 2, 'DegenMoves': 0, 'BarHomogeneous': 1}
    cost_impact, env_impact, water_impact = network.optimization(solver="gurobi", solver_options=solver_options,
                                                                 h2=h2_scenario,
                                                                 h2station=nb_station,
                                                                 sec_base=sector_base, sec_new=sector_new,
                                                                 obj=obj, water=limit_water, ext=extension_production)

    # Plot of the results
    print("INFO: plot of the results...")
    network.plot_network('final', True, False, False)
    enr_inter, operation = network.generator_data()
    if "stock" in h2_scenario:
        network.plot_network('final', False, True, False)
        network.plot_network('final', False, False, True)
        ely, h2stor, fc = network.h2_data(bus=False)
    elif h2_scenario is not None:
        network.plot_network('final', False, True, False)
        ely, h2stor = network.h2_data(bus=True)
