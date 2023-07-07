import os
import pandas as pd
import numpy as np
import functions_used as functions


# Definition of the hydrogen elements : electrolyser, compressor, storage and fuel cell.
# Definition of a hydrogen demand : buses

class H2Chain:
    def __init__(self, data, ps):
        self.storage_data = data["storage"].loc[data["storage"]["technology"] == "h2"]
        self.places = ps
        self.x = data["postes"].loc[data["postes"].index.isin(self.places)]["Long"]
        self.y = data["postes"].loc[data["postes"].index.isin(self.places)]["Lat"]
        self.electrolyser_data = data["link"].loc[data["link"]["technology"] == "electrolyzer"]
        self.compressor_data = data["link"].loc[data["link"]["technology"] == "compressor"]
        self.expander_data = data["link"].loc[data["link"]["technology"] == "expander"]
        self.fc_data = data["link"].loc[data["link"]["technology"] == "fuel cell"]

    def import_electrolyser(self, network, h2size):
        network.madd("Bus",  # PyPSA component
                     "hydrogen bus 30 bar " + self.places,  # Name of the element
                     carrier="hydrogen 30 bar",
                     x=self.x.tolist(),  # Longitude
                     y=self.y.tolist(),  # Latitude
                     )
        if h2size is None:
            network.madd("Link",  # PyPSA component
                         "electrolyser " + self.places,  # Name of the element
                         bus0=("electricity bus " + self.places).tolist(),  # Name of first bus : electricity bus
                         bus1=("hydrogen bus 30 bar " + self.places).tolist(),  # Name of the second bus : hydrogen bus
                         p_nom=0,
                         p_nom_extendable=True,
                         p_nom_min=0,
                         efficiency=self.electrolyser_data["efficiency"].iloc[0],
                         # Efficiency of power transfer from bus0 to bus1
                         capital_cost=functions.calculate_capital_costs(self.electrolyser_data["discount_rate"].iloc[0],
                                                                        self.electrolyser_data["lifetime"].iloc[0],
                                                                        self.electrolyser_data["fixed_OM (%)"].iloc[0],
                                                                        self.electrolyser_data["fixed_OM (tot)"].iloc[
                                                                            0],
                                                                        self.electrolyser_data["CAPEX"].iloc[0], 1),
                         marginal_cost=functions.calculate_marginal_costs(0,
                                                                          self.electrolyser_data["variable_OM"].iloc[0],
                                                                          self.electrolyser_data["efficiency"].iloc[0]),
                         env_f=self.electrolyser_data["env_f"].iloc[0],
                         env_v=self.electrolyser_data["env_v"].iloc[0],
                         water_f=self.electrolyser_data["water_f"].iloc[0],
                         water_v=self.electrolyser_data["water_v"].iloc[0],
                         )
        else:  # TODO en construciton
            network.madd("Link",  # PyPSA component
                         "electrolyser " + self.places,  # Name of the element
                         bus0=("electricity bus " + self.places).tolist(),  # Name of first bus : electricity bus
                         bus1=("hydrogen bus 30 bar " + self.places).tolist(),
                         # Name of the second bus : hydrogen bus
                         p_nom=h2size["electrolyser " + self.places],
                         efficiency=self.electrolyser_data["efficiency"].iloc[0],
                         # Efficiency of power transfer from bus0 to bus1
                         marginal_cost=functions.calculate_marginal_costs(0,
                                                                          self.electrolyser_data["variable_OM"].iloc[0],
                                                                          self.electrolyser_data["efficiency"].iloc[0]),
                         env_f=self.electrolyser_data["env_f"].iloc[0],
                         env_v=self.electrolyser_data["env_v"].iloc[0],
                         water_f=self.electrolyser_data["water_f"].iloc[0],
                         water_v=self.electrolyser_data["water_v"].iloc[0],
                         )

    def import_compressor(self, network):
        network.madd("Bus",  # PyPSA component
                     "hydrogen bus 350 bar " + self.places,  # Name of the element
                     carrier="hydrogen 350 bar",
                     x=self.x.tolist(),  # Longitude
                     y=self.y.tolist()  # Latitude
                     )

        network.madd("Link",  # PyPSA component
                     "compressor " + self.places,  # Name of the element
                     bus0=("hydrogen bus 30 bar " + self.places).tolist(),  # Name of first bus : electricity bus
                     bus1=("hydrogen bus 350 bar " + self.places).tolist(),  # Name of the second bus : hydrogen bus
                     p_nom=0,
                     p_nom_extendable=True,  # Active power which can pass through link is extendable
                     p_nom_min=0,
                     efficiency=self.compressor_data["efficiency"].iloc[0],
                     # Efficiency of power transfer from bus0 to bus1
                     capital_cost=functions.calculate_capital_costs(self.compressor_data["discount_rate"].iloc[0],
                                                                    self.compressor_data["lifetime"].iloc[0],
                                                                    self.compressor_data["fixed_OM (%)"].iloc[0],
                                                                    self.compressor_data["fixed_OM (tot)"].iloc[0],
                                                                    self.compressor_data["CAPEX"].iloc[0], 1),
                     marginal_cost=functions.calculate_marginal_costs(0, self.compressor_data["variable_OM"].iloc[0],
                                                                      self.compressor_data["efficiency"].iloc[0]),
                     env_f=self.compressor_data["env_f"].iloc[0],
                     env_v=self.compressor_data["env_v"].iloc[0],
                     water_f=self.compressor_data["water_f"].iloc[0],
                     water_v=self.compressor_data["water_v"].iloc[0],
                     )

    def import_expander(self, network):  # TODO à enlever probablement
        network.madd("Link",  # PyPSA component
                     "expander " + self.places,  # Name of the element
                     bus0=("hydrogen bus 350 bar " + self.places).tolist(),  # Name of first bus : electricity bus
                     bus1=("hydrogen bus 30 bar " + self.places).tolist(),  # Name of the second bus : hydrogen bus
                     p_nom_extendable=True,  # Active power which can pass through link is extendable
                     efficiency=self.expander_data["efficiency"].iloc[0],
                     # Efficiency of power transfer from bus0 to bus1
                     capital_cost=functions.calculate_capital_costs(self.expander_data["discount_rate"].iloc[0],
                                                                    self.expander_data["lifetime"].iloc[0],
                                                                    self.expander_data["fixed_OM (%)"].iloc[0],
                                                                    self.expander_data["fixed_OM (tot)"].iloc[0],
                                                                    self.expander_data["CAPEX"].iloc[0], 1),
                     marginal_cost=functions.calculate_marginal_costs(0, self.expander_data["variable_OM"].iloc[0],
                                                                      self.expander_data["efficiency"].iloc[0]),
                     env_f=self.expander_data["env_f"].iloc[0],
                     env_v=self.expander_data["env_v"].iloc[0],
                     water_f=self.expander_data["water_f"].iloc[0],
                     water_v=self.expander_data["water_v"].iloc[0],
                     )

    def import_h2_storage_hp(self, network):
        network.madd("Store",  # PyPSA component
                     "hydrogen storage hp " + self.places,  # Name of the element
                     bus=("hydrogen bus 350 bar " + self.places).tolist(),
                     # Name of the bus to which the store is attached
                     carrier=self.storage_data["carrier"].iloc[0],
                     e_nom=0,  # Nominal power (MW)
                     e_nom_extendable=True,  # The capacity can be extended
                     e_nom_min=0,  # Minimum value of capacity
                     e_cyclic=True,
                     capital_cost=functions.calculate_capital_costs(self.storage_data["discount_rate"].iloc[0],
                                                                    self.storage_data["lifetime"].iloc[0],
                                                                    self.storage_data["fixed_OM (%)"].iloc[0],
                                                                    self.storage_data["fixed_OM (tot)"].iloc[0],
                                                                    self.storage_data["CAPEX"].iloc[0],
                                                                    1),  # €/MWh, cost of extending e_nom by 1 MWh
                     marginal_cost=functions.calculate_marginal_costs(self.storage_data["fuel_cost"].iloc[0],
                                                                      self.storage_data["variable_OM"].iloc[0],
                                                                      self.storage_data["efficiency store"].iloc[0] +
                                                                      self.storage_data["efficiency dispatch"].iloc[0]),
                     # marginal cost of the production of 1MWh
                     env_f=self.storage_data["env_f"].iloc[0],
                     env_v=self.storage_data["env_v"].iloc[0],
                     water_f=self.storage_data["water_f"].iloc[0],
                     water_v=self.storage_data["water_v"].iloc[0],
                     )

    def import_h2_storage_lp(self, network, h2size):
        if h2size is None:
            network.madd("Store",  # PyPSA component
                         "hydrogen storage lp " + self.places,  # Name of the element
                         bus=("hydrogen bus 30 bar " + self.places).tolist(),
                         # Name of the bus to which the store is attached
                         e_nom=0,  # Nominal power (MWh)
                         e_nom_extendable=True,  # The capacity can be extended
                         e_nom_min=0,  # Minimum value of capacity
                         e_cyclic=True,
                         capital_cost=functions.calculate_capital_costs(self.storage_data["discount_rate"].iloc[0],
                                                                        self.storage_data["lifetime"].iloc[0],
                                                                        self.storage_data["fixed_OM (%)"].iloc[0],
                                                                        self.storage_data["fixed_OM (tot)"].iloc[0],
                                                                        self.storage_data["CAPEX"].iloc[0],
                                                                        1),  # €/MWh, cost of extending e_nom by 1 MWh
                         marginal_cost=functions.calculate_marginal_costs(self.storage_data["fuel_cost"].iloc[0],
                                                                          self.storage_data["variable_OM"].iloc[0],
                                                                          self.storage_data["efficiency store"].iloc[
                                                                              0] +
                                                                          self.storage_data["efficiency dispatch"].iloc[
                                                                              0]),
                         # marginal cost of the production of 1MWh
                         env_f=self.storage_data["env_f"].iloc[0],
                         env_v=self.storage_data["env_v"].iloc[0],
                         water_f=self.storage_data["water_f"].iloc[0],
                         water_v=self.storage_data["water_v"].iloc[0],
                         )
        else:
            network.madd("Store",  # PyPSA component
                         "hydrogen storage " + self.places,  # Name of the element
                         bus=("hydrogen bus 30 bar " + self.places).tolist(),
                         # Name of the bus to which the store is attached
                         e_nom=h2size["hydrogen storage " + self.places],  # Nominal power (MWh)
                         e_cyclic=True,
                         marginal_cost=functions.calculate_marginal_costs(self.storage_data["fuel_cost"].iloc[0],
                                                                          self.storage_data["variable_OM"].iloc[0],
                                                                          self.storage_data["efficiency store"].iloc[
                                                                              0] +
                                                                          self.storage_data["efficiency dispatch"].iloc[
                                                                              0]),
                         # marginal cost of the production of 1MWh
                         env_f=self.storage_data["env_f"].iloc[0],
                         env_v=self.storage_data["env_v"].iloc[0],
                         water_f=self.storage_data["water_f"].iloc[0],
                         water_v=self.storage_data["water_v"].iloc[0],
                         )

    def import_fc(self, network, h2size):
        if h2size is None:
            network.madd("Link",  # PyPSA component
                         "fuel cell " + self.places,  # Name of the element
                         bus0=("hydrogen bus 30 bar " + self.places).tolist(),  # Name of first bus : electricity bus
                         bus1=("electricity bus " + self.places).tolist(),  # Name of the second bus : hydrogen bus
                         p_nom=0,
                         p_nom_extendable=True,
                         p_nom_min=0,
                         efficiency=self.fc_data["efficiency"].iloc[0],
                         # Efficiency of power transfer from bus0 to bus1
                         capital_cost=functions.calculate_capital_costs(self.fc_data["discount_rate"].iloc[0],
                                                                        self.fc_data["lifetime"].iloc[0],
                                                                        self.fc_data["fixed_OM (%)"].iloc[0],
                                                                        self.fc_data["fixed_OM (tot)"].iloc[0],
                                                                        self.fc_data["CAPEX"].iloc[0], 1),
                         marginal_cost=functions.calculate_marginal_costs(0, self.fc_data["variable_OM"].iloc[0],
                                                                          self.fc_data["efficiency"].iloc[0]),
                         env_f=self.fc_data["env_f"].iloc[0],
                         env_v=self.fc_data["env_v"].iloc[0],
                         water_f=self.fc_data["water_f"].iloc[0],
                         water_v=self.fc_data["water_v"].iloc[0],
                         )
        else:
            network.madd("Link",  # PyPSA component
                         "fuel cell " + self.places,  # Name of the element
                         bus0=("hydrogen bus 30 bar " + self.places).tolist(),  # Name of first bus : electricity bus
                         bus1=("electricity bus " + self.places).tolist(),  # Name of the second bus : hydrogen bus
                         p_nom=h2size["fuel cell " + self.places],
                         efficiency=self.fc_data["efficiency"].iloc[0],
                         # Efficiency of power transfer from bus0 to bus1
                         marginal_cost=functions.calculate_marginal_costs(0, self.fc_data["variable_OM"].iloc[0],
                                                                          self.fc_data["efficiency"].iloc[0]),
                         env_f=self.fc_data["env_f"].iloc[0],
                         env_v=self.fc_data["env_v"].iloc[0],
                         water_f=self.fc_data["water_f"].iloc[0],
                         water_v=self.fc_data["water_v"].iloc[0],
                         )

    def constraint_prodsup_bus(self, n, model, horizon):
        places = self.places.to_xarray()
        # places = places.rename({'Nom du poste source': 'h2buses'})  # TODO est-ce important si on enlève ?

        def electrolyzer_prodsup(m, k):
            """
            Constraint for the electrolyzer to work at least a certain amount of time over the year
            :param m: model
            :param k: station
            :return:
            """
            return sum(m.variables['Link-p'][t, "electrolyser " + k] for t in horizon) - len(horizon) * \
                self.electrolyser_data["capacity factor"].iloc[0] * m.variables['Link-p_nom']["electrolyser " + k] >= 0

        def compressor_prodsup(m, k):
            """
            Constraint for the compressor to work at least a certain amount of time over the year
            :param m: model
            :param k: station
            :return:
            """
            return sum(m.variables['Link-p'][t, "compressor " + k] for t in
                       horizon) - len(horizon) * self.compressor_data["capacity factor"].iloc[0] * \
                m.variables['Link-p_nom']["compressor " + k] >= 0

        model.add_constraints(electrolyzer_prodsup, coords=(places,), name="electrolyzer_prodsup")
        model.add_constraints(compressor_prodsup, coords=(places,), name="compressor_prodsup")


class H2Demand:
    def __init__(self, ps, data_path):
        self.places = ps
        self.data_path = data_path

    def import_h2_bus(self, network, h2bus, nb_disp):
        if os.path.exists(self.data_path + '/conso_hydrogene_' + str(h2bus) + '.csv'):
            data_conso_H2 = pd.read_csv(self.data_path + '/conso_hydrogene_' + str(h2bus) + '.csv', sep=',',
                                        encoding='latin-1',
                                        index_col=0)
            data_conso_H2.index = network.horizon
        # TODO à construire pour réplicabilité
        # else:
        # data_conso_H2 = pd.DataFrame()
        # data_freq = pd.read_excel("U:/PyPSA/Hydrogen (T3)/profil_freq_AB.xlsx", header=1)  # TODO à documenter
        # data_freq.set_index('hour', inplace=True)
        # for i in data_freq.columns[::2]:
        #     df = functions.creation_profil_h2_bus(network.horizon, network.snapshots.normalize().unique(), cons_week=conso_week, cons_sunday=conso_sunday, data=[data_freq[i], data_freq.iloc[:, data_freq.columns.get_indexer([i]) + 1]])
        #     data_conso_H2[i[-17:]] = df

        for i in self.places:
            network.add("Load",  # PyPSA component
                        i + " hydrogen load",  # Name of the element
                        bus="hydrogen bus 350 bar " + i,  # Name of the bus to which load is attached
                        p_set=data_conso_H2[str(nb_disp) + " disp " + str(
                            self.places.size) + " stations"] * 33.33 / self.places.size / 1000,
                        # Active power consumption
                        )

    def import_h2_train(self, network):
        if os.path.exists(self.data_path + '/conso_train.csv'):
            data_conso_H2 = pd.read_csv(self.data_path + '/conso_train.csv', sep=',', encoding='latin-1', index_col=0)
            data_conso_H2.index = network.horizon
            for i in self.places:
                network.add("Load",  # PyPSA component
                            i + " hydrogen load",  # Name of the element
                            bus='hydrogen bus 350 bar ' + i,  # Name of the bus to which load is attached
                            p_set=data_conso_H2.squeeze() * 33.33 / 1000,  # Active power consumption
                            )
