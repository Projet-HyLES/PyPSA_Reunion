import os
import pandas as pd
import numpy as np
import functions_used as functions

# Definition of the electrical demand : base, VE, buses

class ElectricalDemand:
    # TODO classe à remanier pour la rendre plus réplicable
    def __init__(self, network, data, veh, bus):
        self.horizon = network.horizon
        self.days = network.snapshots.normalize().unique()
        self.data_path = network.data_path
        self.ps_transfo = data.loc[data["Transfo"] == "y"].index  # Source substations with transformers

        self.veh_scenario = veh
        self.bus_scenario = bus

    def import_demand(self, network, data, cons):
        data["load"] = pd.DataFrame(0, index=self.horizon, columns=self.ps_transfo)
        data["load_buses"] = data["load_buses"].set_index("Réseau")
        VE = self.import_VE_demand()
        BUS = self.import_bus_demand()

        for i in self.ps_transfo:
            conso = pd.read_csv(network.data_path+"/Consumption "+str(cons)+"/" + i + "_"+str(cons)+".csv", sep=',', encoding='latin-1',
                                index_col=0).squeeze('columns')
            conso.index = self.horizon

            data["load"][i] = conso + VE[i]

            if i in data["load_buses"][network.scenario].values:
                data["load"][i] += BUS[data["load_buses"][data["load_buses"][
                                                              network.scenario] == i].index.values + " " + self.bus_scenario[0]].squeeze() / 1000

            network.add("Load",  # PyPSA component
                        i + " load",  # Name of the element
                        bus="electricity bus " + i,  # Bus to which the demand is attached
                        p_set=data["load"][i],  # Active power consumption
                        )

    def import_VE_demand(self):
        if os.path.exists(self.data_path + "/VE/VE-" + str(self.veh_scenario[0]) + "pilotable-results-" + str(self.veh_scenario[1]) + ".csv"):
            df = pd.read_csv(self.data_path + "/VE/VE-" + str(self.veh_scenario[0]) + "pilotable-results-" + str(self.veh_scenario[1]) + ".csv", sep=',', encoding='latin-1', index_col=0)
            df.index = self.horizon
            return df
        else:
            # Code qui reconstruit la demande, pas forcément nécessaire de le garder
            data = pd.ExcelFile(self.data_path + "/VE/VE-" + str(self.veh_scenario[0]) + "pilotable.xlsx")
            data_parse = {
                "load": data.parse("Load curve"),
                "ratio": data.parse("Demographic distribution"),
                "PS": data.parse("Stations")
            }
            data_parse["load"].set_index('Hour', inplace=True)
            data_parse["ratio"].set_index('Town', inplace=True)
            data_parse["PS"].set_index('Town', inplace=True)

            ps = pd.read_csv(self.data_path + "/postes-sources.csv", sep=';', encoding='latin-1')

            data_parse["load_communes"] = pd.DataFrame(0, index=data_parse["load"].index, columns=data_parse["ratio"].index)
            # Répartition de la courbe de charge journalière par commune
            for i in data_parse["ratio"].index:
                data_parse["load_communes"][i] = data_parse["load"]["MW (316 GWh)"] * \
                                             data_parse["ratio"]["Demographic distribution"].loc[
                                                 data_parse["ratio"].index == i].values

            data_parse["results"] = pd.DataFrame(0, index=self.horizon, columns=ps["Nom du poste source"])
            for i in data_parse["load_communes"].columns:
                alpha = data_parse["PS"].loc[data_parse["PS"].index == i]

                if alpha["S2"].isna().values:
                    to_add = data_parse["load_communes"][i]
                    empty = np.zeros((365, 24))
                    for h in range(empty.shape[1]):
                        empty[:, h] = to_add.iloc[h]

                    empty = empty.reshape((-1, 1))
                    data_parse["results"][alpha["S1"]] += empty

                elif alpha["S3"].isna().values:
                    to_add = data_parse["load_communes"][i] / 2
                    empty = np.zeros((365, 24))
                    for h in range(empty.shape[1]):
                        empty[:, h] = to_add.iloc[h]

                    empty = empty.reshape((-1, 1))
                    data_parse["results"][alpha["S1"]] += empty
                    data_parse["results"][alpha["S2"]] += empty

                else:
                    to_add = data_parse["load_communes"][i] / 3
                    empty = np.zeros((365, 24))
                    for h in range(empty.shape[1]):
                        empty[:, h] = to_add.iloc[h]

                    empty = empty.reshape((-1, 1))
                    data_parse["results"][alpha["S1"]] += empty
                    data_parse["results"][alpha["S2"]] += empty
                    data_parse["results"][alpha["S3"]] += empty

            return data_parse["results"]

    def import_bus_demand(self):
        if os.path.exists(self.data_path+"/conso_bus_urbains.csv"):
            df = pd.read_csv(self.data_path+"/conso_bus_urbains.csv", sep=',', encoding='latin-1', index_col=0)
            df.index = self.horizon
            return df
        else:
            # TODO à construire pour réplicabilité
            profils = pd.read_excel(self.data_path+"/profil_bus_elec.xlsx", header=1, index_col=0)
            data = [profils['week'], profils['Sunday']]
            df = functions.creation_profil_bus(self.horizon, self.days, cons_week, cons_week/2, data)
            return df
