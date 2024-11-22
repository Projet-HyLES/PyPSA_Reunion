import os
import pandas as pd
import functions_used as functions

# Definition of the hydrogen elements : electrolyser, compressor, storage and fuel cell.
# Definition of a hydrogen demand : buses

class H2Chain:
    def __init__(self, data, ps):
        self.storage_data = data["storage"].loc[data["storage"]["technology"] == "h2"]
        self.places = ps
        self.x = data["postes"].loc[data["postes"].index.isin(self.places)]["Long"]
        self.y = data["postes"].loc[data["postes"].index.isin(self.places)]["Lat"]
        self.electrolyser_data = data["link"].loc[data["link"]["technology"] == "electrolyser"]
        self.compressor_data = data["link"].loc[data["link"]["technology"] == "compressor"]
        self.expander_data = data["link"].loc[data["link"]["technology"] == "expander"]
        self.fc_data = data["link"].loc[data["link"]["technology"] == "fuel cell"]

    def import_electrolyser(self, network):
        network.madd("Bus",  # PyPSA component
                     "hydrogen bus 30 bar " + self.places,  # Name of the element
                     carrier="hydrogen 30 bar",
                     x=self.x.tolist(),  # Longitude
                     y=self.y.tolist(),  # Latitude
                     )

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
                                                                    self.electrolyser_data["fixed_OM (tot)"].iloc[0],
                                                                    self.electrolyser_data["CAPEX"].iloc[0], 1),
                     marginal_cost=functions.calculate_marginal_costs(0,
                                                                      self.electrolyser_data["variable_OM"].iloc[0],
                                                                      self.electrolyser_data["efficiency"].iloc[0]),
                     env_f=self.electrolyser_data["env_f"].iloc[0],
                     env_v=self.electrolyser_data["env_v"].iloc[0],
                     water_f=self.electrolyser_data["water_f"].iloc[0],
                     water_v=self.electrolyser_data["water_v"].iloc[0],
                     )

    def import_compressor(self, network):
        for i in self.places:
            if not str("hydrogen bus 350 bar " + i) in network.buses.index:
                network.add("Bus",  # PyPSA component
                            "hydrogen bus 350 bar " + i,  # Name of the element
                            carrier="hydrogen 350 bar",
                            x=network.data["postes"].loc[i]["Long"],  # Longitude
                            y=network.data["postes"].loc[i]["Lat"]  # Latitude
                            )

        network.madd("Link",  # PyPSA component
                     "compressor " + self.places,  # Name of the element
                     bus0=("hydrogen bus 30 bar " + self.places).tolist(),  # Name of first bus : hydrogen bus low pressure
                     bus1=("hydrogen bus 350 bar " + self.places).tolist(),  # Name of second bus : hydrogen bus high pressure
                     bus2=("electricity bus " + self.places).tolist(),  # Name of third bus : electricity bus
                     p_nom=0,
                     p_nom_extendable=True,  # Active power which can pass through link is extendable
                     p_nom_min=0,
                     efficiency=self.compressor_data["efficiency"].iloc[0],
                     efficiency2=self.compressor_data["efficiency2"].iloc[0],
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

    def import_h2_storage_hp(self, network):
        # High pressure means if the present work hydrogen demand. To size high pressure storage, the load is used.
        # (See constraints on the manuscript).
        # Static and series loads are grouped together
        total_load = pd.concat([network.loads.groupby('bus').sum(numeric_only=True).p_set.filter(regex='hydrogen') * 24,
                                network.loads_t.p_set.filter(regex='hydrogen').groupby(pd.to_datetime(network.loads_t.p_set.index).date).sum().max()])
        # Static and series loads are summed when on the same station in order to have a global demand per station
        total_load_total = pd.Series([], dtype=float)
        for i in self.places:
            total_load_total[i] = total_load.filter(regex=i).sum()

        network.madd("Store",  # PyPSA component
                     "hydrogen storage hp " + self.places,  # Name of the element
                     bus=("hydrogen bus 350 bar " + self.places).tolist(),
                     # Name of the bus to which the store is attached
                     carrier=self.storage_data["carrier"].iloc[0],
                     e_nom=0,  # Nominal power (MW)
                     e_cyclic=True,
                     e_nom_extendable=True,  # The capacity can be extended
                     # * 1 : a reserve for several days can be tested
                     e_nom_min=(total_load_total * 1).tolist(),  # Minimum value of capacity
                     capital_cost=functions.calculate_capital_costs(self.storage_data["discount_rate"].iloc[0],
                                                                    self.storage_data["lifetime"].iloc[0],
                                                                    self.storage_data["fixed_OM (%)"].iloc[0],
                                                                    self.storage_data["fixed_OM (tot)"].iloc[0],
                                                                    self.storage_data["CAPEX"].iloc[0],
                                                                    1),  # €/MWh, cost of extending e_nom by 1 MWh
                     marginal_cost=functions.calculate_marginal_costs(self.storage_data["fuel_cost"].iloc[0],
                                                                      self.storage_data["variable_OM"].iloc[0],
                                                                      self.storage_data["efficiency dispatch"].iloc[0]),
                     # marginal cost of the production of 1MWh
                     env_f=self.storage_data["env_f"].iloc[0],
                     env_v=self.storage_data["env_v"].iloc[0],
                     water_f=self.storage_data["water_f"].iloc[0],
                     water_v=self.storage_data["water_v"].iloc[0],
                     )

    def import_h2_storage_lp(self, network):
        network.madd("Store",  # PyPSA component
                     "hydrogen storage lp " + self.places,  # Name of the element
                     bus=("hydrogen bus 30 bar " + self.places).tolist(),
                     # Name of the bus to which the store is attached
                     e_nom=0,  # Nominal power (MWh)
                     e_nom_extendable=True,  # The capacity can be extended
                     e_nom_min=0,  # Minimum value of capacity
                     e_cyclic=False,
                     capital_cost=functions.calculate_capital_costs(self.storage_data["discount_rate"].iloc[0],
                                                                    self.storage_data["lifetime"].iloc[0],
                                                                    self.storage_data["fixed_OM (%)"].iloc[0],
                                                                    self.storage_data["fixed_OM (tot)"].iloc[0],
                                                                    self.storage_data["CAPEX"].iloc[0],
                                                                    1),  # €/MWh, cost of extending e_nom by 1 MWh
                     marginal_cost=functions.calculate_marginal_costs(self.storage_data["fuel_cost"].iloc[0],
                                                                      self.storage_data["variable_OM"].iloc[0],
                                                                      self.storage_data["efficiency dispatch"].iloc[0]),
                     # marginal cost of the production of 1MWh
                     env_f=self.storage_data["env_f"].iloc[0],
                     env_v=self.storage_data["env_v"].iloc[0],
                     water_f=self.storage_data["water_f"].iloc[0],
                     water_v=self.storage_data["water_v"].iloc[0],
                     )

    def import_fc(self, network):
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

    def import_expander(self, network):
        network.madd("Link",  # PyPSA component
                     "expander " + self.places,  # Name of the element
                     bus0=("hydrogen bus 350 bar " + self.places).tolist(),  # Name of first bus : electricity bus
                     bus1=("hydrogen bus 30 bar " + self.places).tolist(),  # Name of the second bus : hydrogen bus
                     p_nom=0,
                     p_nom_extendable=True,
                     p_nom_min=0,
                     )

    def constraint_prodsup_bus(self, model, horizon):
        places = self.places.to_xarray()

        def electrolyser_prodsup(m, k):
            """
            Constraint for the electrolyser to work at least a certain amount of time over the year
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

        model.add_constraints(electrolyser_prodsup, coords=(places,), name="electrolyser_prodsup")
        model.add_constraints(compressor_prodsup, coords=(places,), name="compressor_prodsup")

    def constraint_cyclic_soc(self, model, horizon, p):
        # p is the permitted parameter in % for deviating from the strict cyclic condition
        places = self.places.to_xarray()
        
        def cyclic_inf(m, k):
            """
            Constraint for the initial state of charge of the storage to be below the final state of charge, times a
            certain percentage.
            :param m: model
            :param k: station
            :return:
            """
            return m.variables['Store-e'][horizon[-1], "hydrogen storage hp " + k] -\
                m.variables['Store-e'][horizon[0], "hydrogen storage hp " + k] * (1 + p/100) <= 0
            
        def cyclic_sup(m, k):
            """
            Constraint for the initial state of charge of the storage to be above the final state of charge, times a
            certain percentage.
            :param m: model
            :param k: station
            :return:
            """
            return m.variables['Store-e'][horizon[-1], "hydrogen storage hp " + k] - \
                m.variables['Store-e'][horizon[0], "hydrogen storage hp " + k] * (1 - p / 100) >= 0
            
        model.add_constraints(cyclic_inf, coords=(places,), name="cyclic_inf")
        model.add_constraints(cyclic_sup, coords=(places,), name="cyclic_sup")

    def constraint_minimal_soc(self, n, model, horizon, l, aviation, maritime):
        places = self.places.to_xarray()
        snap = pd.Series(horizon).to_xarray().rename({'index': 'snapshots'})

        total_load = pd.concat(
            [n.loads.groupby('bus').sum(numeric_only=True).p_set.filter(regex='hydrogen') * 24,
             n.loads_t.p_set.filter(regex='hydrogen').groupby(
                 pd.to_datetime(n.loads_t.p_set.index).date).sum().max()])
        if aviation:
            total_load = total_load.filter(regex="Aviation")
            def soc_min(m, i, j):
                """
                Constraint for the initial state of charge of the storage to be below the final state of charge, times a
                certain percentage.
                :param m: model
                :param i: station
                :param j: snapshot
                :return:
                """
                return m.variables['Store-e'][j, "hydrogen storage hp " + i] >= l * total_load.values[0]

            model.add_constraints(soc_min, coords=(places, snap), name="soc_min")
        if maritime:
            total_load = total_load.filter(regex="Maritime")
            def soc_min(m, i, j):
                """
                Constraint for the initial state of charge of the storage to be below the final state of charge, times a
                certain percentage.
                :param m: model
                :param i: station
                :param j: snapshot
                :return:
                """
                return m.variables['Store-e'][j, "hydrogen storage hp " + i] >= l * total_load.values[0]

            model.add_constraints(soc_min, coords=(places, snap), name="soc_min")
        if not maritime and not aviation:
            # Static and series loads are summed when on the same station in order to have a global demand per station
            total_load_total = pd.Series([], dtype=float)
            for i in self.places:
                total_load_total[i] = total_load.filter(regex=i).sum()

            def soc_min(m, i, j):
                """
                Constraint for the initial state of charge of the storage to be below the final state of charge, times a
                certain percentage.
                :param m: model
                :param i: station
                :param j: snapshot
                :return:
                """
                return m.variables['Store-e'][j, "hydrogen storage hp " + i] >= l * total_load_total[i]

            model.add_constraints(soc_min, coords=(places, snap), name="soc_min")

class H2Demand:
    def __init__(self, ps, data_path):
        self.places = ps
        self.data_path = data_path

    def import_h2_buses(self, network, h2bus, nb_disp):
        if os.path.exists(self.data_path + '/conso_hydrogene_' + str(h2bus) + '.csv'):
            data_conso_h2 = pd.read_csv(self.data_path + '/conso_hydrogene_' + str(h2bus) + '_' + str(network.snapshots.year.unique().values[0]) + '.csv', sep=',',
                                        encoding='latin-1',
                                        index_col=0)
            data_conso_h2.index = network.horizon
        else:
            raise ValueError('ERROR: no demand for buses.')

        for i in self.places:
            if not str("hydrogen bus 350 bar " + i) in network.buses.index:
                network.add("Bus",  # PyPSA component
                            "hydrogen bus 350 bar " + i,  # Name of the element
                            carrier="hydrogen 350 bar",
                            x=network.data["postes"].loc[i]["Long"],  # Longitude
                            y=network.data["postes"].loc[i]["Lat"]  # Latitude
                            )

            network.add("Load",  # PyPSA component
                        i + " hydrogen buses load",  # Name of the element
                        bus="hydrogen bus 350 bar " + i,  # Name of the bus to which load is attached
                        p_set=data_conso_h2[str(nb_disp) + " disp " + str(
                            self.places.size) + " stations"] * 33.33 / self.places.size / 1000,
                        # Active power consumption
                        )

    def import_h2_train(self, network):
        if os.path.exists(self.data_path + '/conso_train.csv'):
            data_conso_h2 = pd.read_csv(self.data_path + '/conso_train.csv', sep=',', encoding='latin-1', index_col=0)
            data_conso_h2.index = network.horizon
        else:
            raise ValueError('ERROR: no demand for train.')

        for i in self.places:
            if not str("hydrogen bus 350 bar " + i) in network.buses.index:
                network.add("Bus",  # PyPSA component
                            "hydrogen bus 350 bar " + i,  # Name of the element
                            carrier="hydrogen 350 bar",
                            x=network.data["postes"].loc[i]["Long"],  # Longitude
                            y=network.data["postes"].loc[i]["Lat"]  # Latitude
                            )

            network.add("Load",  # PyPSA component
                        i + " hydrogen train load",  # Name of the element
                        bus='hydrogen bus 350 bar ' + i,  # Name of the bus to which load is attached
                        p_set=data_conso_h2[i].squeeze() * 33.33 / 1000,  # Active power consumption
                        )


    def import_aviation_hydrogen_demand(self, network, rt):
        network.add("Bus",  # PyPSA component
                    "hydrogen bus 350 bar Roland Garros airport",  # Name of the element
                    carrier="hydrogen 350 bar",
                    x=network.buses.x['electricity bus Roland Garros airport'],  # Longitude
                    y=network.buses.y['electricity bus Roland Garros airport']  # Latitude
                    )

        if rt is None:
            if os.path.exists(self.data_path + "/demand_aviation_h2.csv"):
                conso = pd.read_csv(self.data_path + "/demand_aviation_h2.csv", sep=',', encoding='latin-1', index_col=0).squeeze('columns')
                conso.index = network.horizon

                network.add("Load",  # PyPSA component
                            "Aviation hydrogen load",  # Name of the element
                            bus="hydrogen bus 350 bar Roland Garros airport",  # Bus to which the demand is attached
                            p_set=conso,  # Active power consumption
                            )
            else:
                raise ValueError('ERROR: no aircraft hydrogen demand file found.')
        else:
            network.add("Generator",  # PyPSA component
                        "Aviation hydrogen export",  # Name of the element
                        bus="hydrogen bus 350 bar Roland Garros airport",  # Bus to which the demand is attached
                        carrier='joker',
                        p_nom_extendable=True,
                        p_min_pu=-rt['Ratio'] * 0.90103,
                        p_max_pu=-rt['Ratio'] * 0.90103,
                        marginal_cost=10000
                        )

        if not str("hydrogen bus 30 bar Roland Garros airport") in network.buses.index:
            network.add("Bus",  # PyPSA component
                        "hydrogen bus 30 bar Roland Garros airport",  # Name of the element
                        carrier="hydrogen 30 bar",
                        x=network.buses.x['electricity bus Roland Garros airport'],  # Longitude
                        y=network.buses.y['electricity bus Roland Garros airport'],  # Latitude
                        )
            network.add("Link",  # PyPSA component
                        "electrolyser Roland Garros airport" + self.places,
                        bus0="electricity bus Roland Garros airport",
                        bus1="hydrogen bus 30 bar Roland Garros airport",
                        p_nom=0,
                        p_nom_extendable=True,
                        p_nom_min=0,
                        efficiency=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["efficiency"].iloc[0],
                        capital_cost=functions.calculate_capital_costs(network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["discount_rate"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["lifetime"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["fixed_OM (%)"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["fixed_OM (tot)"].iloc[
                                                                           0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["CAPEX"].iloc[0], 1),
                        marginal_cost=functions.calculate_marginal_costs(0,
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["variable_OM"].iloc[0],
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["efficiency"].iloc[0]),
                        env_f=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["env_f"].iloc[0],
                        env_v=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["env_v"].iloc[0],
                        water_f=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["water_f"].iloc[0],
                        water_v=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["water_v"].iloc[0],
                        )
            network.add("Link",  # PyPSA component
                        "compressor Roland Garros airport",  # Name of the element
                        bus0="hydrogen bus 30 bar Roland Garros airport",
                        bus1="hydrogen bus 350 bar Roland Garros airport",
                        bus2="electricity bus Roland Garros airport",  # Name of third bus : electricity bus
                        p_nom=0,
                        p_nom_extendable=True,  # Active power which can pass through link is extendable
                        p_nom_min=0,
                        efficiency=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["efficiency"].iloc[0],
                        efficiency2=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["efficiency2"].iloc[0],
                        capital_cost=functions.calculate_capital_costs(network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["discount_rate"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["lifetime"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["fixed_OM (%)"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["fixed_OM (tot)"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["CAPEX"].iloc[0], 1),
                        marginal_cost=functions.calculate_marginal_costs(0,
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["variable_OM"].iloc[0],
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["efficiency"].iloc[0]),
                        env_f=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["env_f"].iloc[0],
                        env_v=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["env_v"].iloc[0],
                        water_f=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["water_f"].iloc[0],
                        water_v=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["water_v"].iloc[0],
                        )
            network.add("Store",  # PyPSA component
                        "hydrogen storage hp Roland Garros airport",  # Name of the element
                        bus="hydrogen bus 350 bar Roland Garros airport",
                        carrier=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["carrier"].iloc[0],
                        e_nom=0,  # Nominal power (MW)
                        e_cyclic=True,
                        e_nom_extendable=True,  # The capacity can be extended
                        e_nom_min=0,  # Minimum value of capacity
                        capital_cost=functions.calculate_capital_costs(network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["discount_rate"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["lifetime"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["fixed_OM (%)"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["fixed_OM (tot)"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["CAPEX"].iloc[0],
                                                                       1),  # €/MWh, cost of extending e_nom by 1 MWh
                        marginal_cost=functions.calculate_marginal_costs(network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["fuel_cost"].iloc[0],
                                                                         network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["variable_OM"].iloc[0],
                                                                         network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["efficiency dispatch"].iloc[0]),
                        # marginal cost of the production of 1MWh
                        env_f=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["env_f"].iloc[0],
                        env_v=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["env_v"].iloc[0],
                        water_f=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["water_f"].iloc[0],
                        water_v=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["water_v"].iloc[0],
                        )



    def import_maritime_hydrogen_demand(self, network):
        if os.path.exists(self.data_path + "/demand_maritime_h2.csv"):
            conso = pd.read_csv(self.data_path + "/demand_maritime_h2.csv", sep=',', encoding='latin-1', index_col=0).squeeze('columns')
            conso.index = network.horizon
        else:
            raise ValueError('ERROR: no maritim hydrogen demand file found.')

        if not str("hydrogen bus 350 bar Marquet") in network.buses.index:
            network.add("Bus",  # PyPSA component
                        "hydrogen bus 350 bar Marquet",  # Name of the element
                        carrier="hydrogen 350 bar",
                        x=network.buses.x['electricity bus Marquet'],  # Longitude
                        y=network.buses.y['electricity bus Marquet']  # Latitude
                        )

        network.add("Load",  # PyPSA component
                    "Maritime hydrogen load",  # Name of the element
                    bus="hydrogen bus 350 bar Marquet",  # Bus to which the demand is attached
                    p_set=conso*0.75,  # Active power consumption
                    )

        if not str("hydrogen bus 30 bar Marquet") in network.buses.index:
            network.add("Bus",  # PyPSA component
                        "hydrogen bus 30 bar Marquet",  # Name of the element
                        carrier="hydrogen 30 bar",
                        x=network.buses.x['electricity bus Marquet'],  # Longitude
                        y=network.buses.y['electricity bus Marquet'],  # Latitude
                        )
            network.add("Link",  # PyPSA component
                        "electrolyser Marquet" + self.places,
                        bus0="electricity bus Marquet",
                        bus1="hydrogen bus 30 bar Marquet",
                        p_nom=0,
                        p_nom_extendable=True,
                        p_nom_min=0,
                        efficiency=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["efficiency"].iloc[0],
                        capital_cost=functions.calculate_capital_costs(network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["discount_rate"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["lifetime"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["fixed_OM (%)"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["fixed_OM (tot)"].iloc[
                                                                           0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["CAPEX"].iloc[0], 1),
                        marginal_cost=functions.calculate_marginal_costs(0,
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["variable_OM"].iloc[0],
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["efficiency"].iloc[0]),
                        env_f=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["env_f"].iloc[0],
                        env_v=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["env_v"].iloc[0],
                        water_f=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["water_f"].iloc[0],
                        water_v=network.data["link"].loc[network.data["link"]["technology"] == "electrolyser"]["water_v"].iloc[0],
                        )
            network.add("Link",  # PyPSA component
                        "compressor Marquet",  # Name of the element
                        bus0="hydrogen bus 30 bar Marquet",
                        bus1="hydrogen bus 350 bar Marquet",
                        bus2="electricity bus Marquet",  # Name of third bus : electricity bus
                        p_nom=0,
                        p_nom_extendable=True,  # Active power which can pass through link is extendable
                        p_nom_min=0,
                        efficiency=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["efficiency"].iloc[0],
                        efficiency2=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["efficiency2"].iloc[0],
                        capital_cost=functions.calculate_capital_costs(network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["discount_rate"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["lifetime"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["fixed_OM (%)"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["fixed_OM (tot)"].iloc[0],
                                                                       network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["CAPEX"].iloc[0], 1),
                        marginal_cost=functions.calculate_marginal_costs(0,
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["variable_OM"].iloc[0],
                                                                         network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["efficiency"].iloc[0]),
                        env_f=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["env_f"].iloc[0],
                        env_v=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["env_v"].iloc[0],
                        water_f=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["water_f"].iloc[0],
                        water_v=network.data["link"].loc[network.data["link"]["technology"] == "compressor"]["water_v"].iloc[0],
                        )
            total_load = pd.concat(
                [network.loads.groupby('bus').sum(numeric_only=True).p_set.filter(regex='hydrogen') * 24,
                 network.loads_t.p_set.filter(regex='hydrogen').groupby(
                     pd.to_datetime(network.loads_t.p_set.index).date).sum().max()])
            network.add("Store",  # PyPSA component
                        "hydrogen storage hp Marquet",  # Name of the element
                        bus="hydrogen bus 350 bar Marquet",
                        carrier=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["carrier"].iloc[0],
                        e_nom=0,  # Nominal power (MW)
                        e_cyclic=True,
                        e_nom_extendable=True,  # The capacity can be extended
                        e_nom_min=total_load.filter(regex="Maritime"),  # Minimum value of capacity
                        capital_cost=functions.calculate_capital_costs(network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["discount_rate"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["lifetime"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["fixed_OM (%)"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["fixed_OM (tot)"].iloc[0],
                                                                       network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["CAPEX"].iloc[0],
                                                                       1),  # €/MWh, cost of extending e_nom by 1 MWh
                        marginal_cost=functions.calculate_marginal_costs(network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["fuel_cost"].iloc[0],
                                                                         network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["variable_OM"].iloc[0],
                                                                         network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["efficiency dispatch"].iloc[0]),
                        # marginal cost of the production of 1MWh
                        env_f=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["env_f"].iloc[0],
                        env_v=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["env_v"].iloc[0],
                        water_f=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["water_f"].iloc[0],
                        water_v=network.data["storage"].loc[network.data["storage"]["technology"] == "h2"]["water_v"].iloc[0],
                        )
