import os
import pandas as pd
import numpy as np

# Definition of the electrical demand : base, electric vehicles, buses

class ElectricalDemand:
    def __init__(self, network):
        self.network = network
        self.nb_year = len(self.network.snapshots.year.unique().values)
        self.horizon = network.horizon
        self.days = network.snapshots.normalize().unique()
        self.data_dir = network.data_dir
        self.ps_transfo = self.network.data["postes"].loc[self.network.data["postes"]["Transfo"] == "y"].index  # Substations with a transformer
        self.veh_scenario = self.network.vehicles_scenario
        self.bus_scenario = self.network.buses_scenario

    def import_demand(self, multiyear):
        self.network.data["load"] = pd.DataFrame(0, index=self.horizon, columns=self.ps_transfo)
        self.network.data["load_buses"] = self.network.data["load_buses"].set_index("Réseau")
        year = self.network.year
        elec_veh = self.import_elec_veh_demand(multiyear)
        buses = self.import_bus_demand(multiyear)

        conso = pd.read_csv(self.data_dir+"/Consumption "+str(self.network.cons)+"/" + str(year) + "_" + str(self.network.climate_scenario)+".csv",
                            sep=',', encoding='latin-1', index_col=0).squeeze('columns')
        if multiyear:
            conso = pd.concat([conso] * self.nb_year, ignore_index=True)  # No consumption data for several year
        conso.index = self.horizon

        for i in self.ps_transfo:
            self.network.data["load"][i] = conso[i] + elec_veh[i]

            if i in self.network.data["load_buses"][self.network.scenario].values:
                self.network.data["load"][i] += buses[self.network.data["load_buses"][self.network.data["load_buses"][
                                                             self.network.scenario] == i].index.values + " " + self.bus_scenario[0]].squeeze() / 1000

            self.network.add("Load",  # PyPSA component
                             i + " load",  # Name of the element
                             bus="electricity bus " + i,  # Bus to which the demand is attached
                             p_set=self.network.data["load"][i],  # Active power consumption
                             )

    def import_elec_veh_demand(self, multiyear):
        if os.path.exists(self.data_dir + "/VE/VE-" + str(self.veh_scenario[0]) + "pilotable-results-" + str(self.veh_scenario[1]) + ".csv"):
            df = pd.read_csv(self.data_dir + "/VE/VE-" + str(self.veh_scenario[0]) + "pilotable-results-" + str(self.veh_scenario[1]) + ".csv", sep=',', encoding='latin-1', index_col=0)
            if multiyear:
                df = pd.concat([df] * self.nb_year, ignore_index=True)  # No consumption data for several year
            df.index = self.horizon
            return df
        else:
            raise ValueError('ERROR: no electric vehicles file found.')

    def import_bus_demand(self, multiyear):
        if os.path.exists(self.data_dir+"/conso_bus_urbains.csv"):
            df = pd.read_csv(self.data_dir+"/conso_bus_urbains.csv", sep=',', encoding='latin-1', index_col=0)
            if multiyear:
                df = pd.concat([df] * self.nb_year, ignore_index=True)  # No consumption data for several year
            df.index = self.horizon
            return df
        else:
            raise ValueError('ERROR: no electric buses file found.')

    def import_aircraft_elec_demand(self, rt):
        if rt is None:  # no optimization of aircraft fuels produced locally
            if os.path.exists(self.data_dir + "/demand_aviation_elec.csv"):
                conso = pd.read_csv(self.data_dir + "/demand_aviation_elec.csv", sep=',', encoding='latin-1', index_col=0).squeeze('columns')
                conso.index = self.horizon
                self.network.add("Load",  # PyPSA component
                                 "Aviation electricity load",  # Name of the element
                                 bus="electricity bus Roland Garros airport",  # Bus to which the demand is attached
                                 p_set=conso,  # Active power consumption
                                 )
            else:
                raise ValueError('ERROR: no aircraft electricity demand file found.')
        else:
            self.network.add("Generator",  # PyPSA component
                             "Aviation electricity export",  # Name of the element
                             bus="electricity bus Roland Garros airport",  # Bus to which the demand is attached
                             carrier='aircraft',
                             p_nom_extendable=True,
                             p_min_pu=-rt['Ratio'] * 0.09897,
                             p_max_pu=-rt['Ratio'] * 0.09897,
                             marginal_cost=1000
                             )

    def import_maritime_elec_demand(self):
        if os.path.exists(self.data_dir + "/demand_maritime_elec.csv"):
            conso = pd.read_csv(self.data_dir + "/demand_maritime_elec.csv", sep=',', encoding='latin-1', index_col=0).squeeze('columns')
            conso.index = self.horizon
            self.network.add("Load",  # PyPSA component
                             "Maritime electricity load",  # Name of the element
                             bus="electricity bus Marquet",  # Bus to which the demand is attached
                             p_set=conso*0.75,  # Active power consumption : times 0.75 to consider reduction of
                             # maritime traffic for Reunion island (25% of maritime traffic is currently for fossil fuels)
                             )
        else:
            raise ValueError('ERROR: no maritimz electricity demand file found.')
