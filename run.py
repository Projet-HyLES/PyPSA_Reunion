import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from energy_network import EnergyNetwork
import matplotlib

matplotlib.use('TkAgg')  # can be reomved if matplotlib figures are plotted automatically

if __name__ == '__main__':
    """
    This script initializes the simulation, imports the network data, and plots the initial network.
    """

    # pd.set_option("display.max_rows", None, "display.max_columns", None)  # activate to print every row of a dataframe

    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.")

    # Initialization of the simulation
    multiyear = False  # set to True if multiple years are being run
    if multiyear:
        year1 = 2005
        year2 = 2009
        year_data = 2050  # technologies data are only provided for 2030 or 2050
        snapshots = pd.date_range(f"{year1}-01-01 00:00", f"{year2}-12-31 23:00", freq="h")
    else:
        year = 2050
        year_data = 0
        snapshots = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")

    snapshots = snapshots[(snapshots.month != 2) | (snapshots.day != 29)]  # 29th of february is removed for simulations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'Data')

    h2_scenario = 'None'  # ['stock', 'buses', 'stock+buses', 'train', 'train+buses', 'stock+buses+train', 'None']
    h2_bus_scenario = None  # bus frequency scenario (influences hydrogen demand)
    nb_station = None  # number of hydrogen refueling stations for buses scenario
    nb_disp = None  # number of dispensers per refueling station for buses scenario
    if "buses" in h2_scenario:
        h2_bus_scenario = "freqA"  # freqA, freqB
        nb_station = 2  # 2 ou 3
        nb_disp = 3  # 1, 2 ou 3 par station
    extension_production = False  # if True, the capacity of some generators is extendable

    selfsufficiency = False  # if True, total quantity of aircraft fuels produced locally will be optimised
    if selfsufficiency:
        aircraft = True
        marine = True
    else:
        aircraft = False  # set to True to consider aircraft fuels
        marine = False  # set to True to consider marine fuels

    # Import of the network
    tic = time.time()
    network = EnergyNetwork(snapshots)
    sector_base, sector_new = network.import_network(data_dir,
                                                     h2=h2_scenario, h2bus=h2_bus_scenario,
                                                     h2station=nb_station, h2disp=nb_disp,
                                                     ext=extension_production,
                                                     selfsufficiency=selfsufficiency,
                                                     aircraft=aircraft,
                                                     marine=marine,
                                                     multiyear=multiyear,
                                                     yeardata=year_data)
    toc = time.time()
    print("INFO: Importing data took {} seconds.".format(toc - tic))

    # raise ValueError('ERROR: STOP.')

    network.plot_network('initial', False, False, False)

    # Optimization of the system
    obj = 'cost'  # cost, env, multi
    limit_water = None  # constraint to limit total water consumed

    solver_options = {'Method': 2, 'DegenMoves': 0, 'BarHomogeneous': 1}
    if obj == 'multi':
        cost_list, env_list = network.optimization(solver="gurobi", solver_options=solver_options,
                                                   h2=h2_scenario, sec_new=sector_new,
                                                   obj=obj, water=limit_water,
                                                   ext=extension_production, multiyear=multiyear,
                                                   selfsufficiency=selfsufficiency)
        fig = plt.figure()
        plt.scatter(cost_list, env_list)
        plt.title('Pareto front of the simulated system')
        plt.xlabel('Costs (€)')
        plt.ylabel('Environmental impact (?)')
        fig.tight_layout()
        fig.savefig("pareto_front.pdf", bbox_inches="tight", dpi=300)

    else:
        cost_impact, env_impact, water_impact = network.optimization(solver="gurobi", solver_options=solver_options,
                                                                     h2=h2_scenario, sec_new=sector_new,
                                                                     obj=obj, water=limit_water,
                                                                     ext=extension_production, multiyear=multiyear,
                                                                     selfsufficiency=selfsufficiency)

    # Plot of the results
    print("INFO: plot of the results...")
    network.plot_network('final', True, False, False)
    enr_inter, operation = network.generator_data()
    if "stock" in h2_scenario:
        network.plot_network('final', False, True, False)
        network.plot_network('final', False, False, True)
        ely, h2stor, fc = network.h2_data(bus=False)
    elif h2_scenario != 'None':
        network.plot_network('final', False, True, False)
        ely, h2stor = network.h2_data(bus=True)
