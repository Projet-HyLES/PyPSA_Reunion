import time
import pathlib
import pandas as pd
from tictoc import tic, toc
from energy_network import EnergyNetwork

if __name__ == '__main__':
    # pd.set_option("display.max_rows", None, "display.max_columns", None)  # activate to print every row of a dataframe
    print("Start time: {}.".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    # Initialization of the simulation
    current_path = pathlib.Path(__file__).parent.resolve()
    data_path = str(current_path)+'/Data'
    snapshots = pd.date_range("2050-01-01 00:00", "2050-12-31 23:00", freq="H")  # 2050 simulation
    # snapshots = pd.date_range("2030-01-01 00:00", "2030-12-31 23:00", freq="H")  # 2030 simulation

    h2_scenario = None  # stock, bus, stock+bus, stock+hysteresis, train, train+bus, stock+bus+train, None

    h2_installations = None
    h2_bus_scenario = None
    nb_station = None
    nb_disp = None
    stations = {}
    # if "bus" in h2_scenario:
    #     h2_bus_scenario = "freqA"  # freqA, freqB
    #     nb_station = 3  # 2 ou 3
    #     nb_disp = 1  # 1, 2 ou 3 par station

    extension_production = False  # if True, the capacity of some generators is extendable (TODO à conserver ou jeter ?)

    # Import of the network
    tic()
    network = EnergyNetwork(snapshots)
    sector_base, sector_new = network.import_network(data_path, h2=h2_scenario, h2bus=h2_bus_scenario,
                                                     h2disp=nb_disp, h2size=h2_installations, ext=extension_production)
    t = toc(False)
    print("INFO: importing data took {} seconds.".format(t))

    network.plot_network('initial', False, False, False)

    # Optimization of the system
    obj = 'cost'  # cost, env, multi
    limit_water = None

    tic()
    solver_options = {'Method': 3, 'DegenMoves': 0, 'BarHomogeneous': 1}
    cost_impact, env_impact, water_impact = network.optimization(solver="gurobi", solver_options=solver_options, h2=h2_scenario,
                                                                 sec_base=sector_base, sec_new=sector_new,
                                                                 obj=obj, water=limit_water, ext=extension_production)
    t = toc(False)
    print("INFO: solving took {} hours.".format(t))

    # Plot of the results
    print("INFO: plot of the results...")
    network.plot_network('final', True, False, False)
    enr_inter = network.generator_data()
    if (h2_scenario == "train") or (h2_scenario == "bus"):
        network.plot_network('final', False, True, False)
        ely, h2stor = network.h2_data(bus=True)
    elif h2_scenario is not None:
        network.plot_network('final', False, True, False)
        network.plot_network('final', False, False, True)
        ely, h2stor, fc = network.h2_data(bus=False)
