import re
import pandas as pd
import functions_used as functions


# Definition of the electrical grid : substations, electrical lines, existing batteries and additional batteries.


class ElectricalGrid:
    def __init__(self, data, lines):
        self.buses_name = data["postes"].index
        self.buses_vnom = data["postes"]["Voltage"].tolist()
        self.buses_x = data["postes"]["Long"].tolist()
        self.buses_y = data["postes"]["Lat"].tolist()
        self.lines = lines
        self.r = 0.06
        self.x = 0.37
        self.capcostlines = 10400 + 25000  # TODO à update fonction de la puissance (pb linéarité ?)
        self.islandfact = 1.2
        self.env_f = 1
        self.env_v = 1
        self.water_f = 1
        self.water_v = 1

    def import_buses(self, network):
        network.madd("Bus",  # PyPSA component
                     "electricity bus " + self.buses_name,  # Name of the element
                     v_nom=self.buses_vnom,  # Nominal voltage
                     x=self.buses_x,  # Longitude
                     y=self.buses_y,  # Latitude
                     )

    def import_lines(self, network):
        pattern = r'[0-9]'
        for index, row in self.lines.iterrows():
            if row.eq("Piquage").any():
                print("INFO: line {} is incomplete and joins another one.".format(row[
                                                                                      "Nom de la ligne"]))  # the line can be ignored, the substation is linked to the other substations with a line "well named"
                continue
            if len(row["Nom de la ligne"].split('/')) > 2:  # One line can link more than two substations
                for i in range(len(row["Nom de la ligne"].split('/')) - 1):
                    for j in range(i + 1, len(row["Nom de la ligne"].split('/'))):
                        new_name0 = re.sub(pattern, '', row["bus" + str(i)])
                        new_name1 = re.sub(pattern, '', row["bus" + str(j)])
                        network.add("Line",  # PyPSA component
                                    row["bus" + str(i)] + " to " + row["bus" + str(j)] + " " + row["Type"],
                                    # Name of the element
                                    bus0="electricity bus " + new_name0,  # First bus to which branch is attached
                                    bus1="electricity bus " + new_name1,  # Second bus to which branch is attached
                                    s_nom=row["Capacite (MVA)"],  # Limit of apparent power (MVA)
                                    s_nom_extendable=True,  # Capacity is allowed to be extended
                                    s_nom_min=row["Capacite (MVA)"],  # Minimum value of capacity
                                    x=self.x,  # Reactance, source : cours de Robin
                                    r=self.r,
                                    capital_cost=self.capcostlines * row["Longueur (km)"] * self.islandfact,
                                    # €/MVA, cost for extending s_nom by 1 MVA
                                    )
            else:
                new_name0 = re.sub(pattern, '', row["bus0"])
                new_name1 = re.sub(pattern, '', row["bus1"])
                network.add("Line",  # PyPSA component
                            row["bus0"] + " to " + row["bus1"] + " " + row["Type"],  # Name of the lement
                            bus0="electricity bus " + new_name0,  # First bus to which branch is attached
                            bus1="electricity bus " + new_name1,  # Second bus to which branch is attached
                            s_nom=row["Capacite (MVA)"],  # Limit of apparent power (MVA)
                            s_nom_extendable=True,  # Capacity is allowed to be extended
                            s_nom_min=row["Capacite (MVA)"],  # Minimum value of capacity
                            x=self.x,  # Reactance, source : cours de Robin
                            r=self.r,
                            capital_cost=self.capcostlines * row["Longueur (km)"] * self.islandfact,
                            # €/MVA, cost for extending s_nom by 1 MVA (source rapport ADEME)
                            )


class ExistingStorages:
    def __init__(self, data):
        self.data = data["batteries"]
        self.kind = self.data["kind"]
        self.places = self.data.index
        self.x = data["postes"].loc[data["postes"].index.isin(self.places.tolist())]["Long"]
        self.y = data["postes"].loc[data["postes"].index.isin(self.places.tolist())]["Lat"]
        self.fuelcost = self.data["fuel_cost"]
        self.variableom = self.data["variable_OM"]
        self.efficiencystore = self.data["efficiency store"]
        self.efficiencydispatch = self.data["efficiency dispatch"]
        self.standingloss = self.data["standing loss"]
        self.capacity = self.data["capacity"]
        self.eminpu = self.data["soc min"]
        self.emaxpu = self.data["soc max"]
        self.env_f = self.data["env_f"]
        self.env_v = self.data["env_v"]
        self.water_f = self.data["water_f"]
        self.water_v = self.data["water_v"]  # TODO vraiment utile si y a déjà self.data de défini ?

    def import_storages(self, network):
        for i in self.places:
            if self.kind[i] == "power":
                network.add("StorageUnit",  # PyPSA component
                            "existing battery " + i,  # Name of the element
                            bus="electricity bus " + i,  # Name of the bus to which the storage is attached
                            p_nom=self.capacity[i],  # Nominal power (MW)
                            marginal_cost=functions.calculate_marginal_costs(self.fuelcost[i], self.variableom[i],
                                                                             self.efficiencystore[i] *
                                                                             self.efficiencydispatch[i]),
                            # Marginal cost of the production of 1MWh
                            cyclic_state_of_charge=True,
                            efficiency_store=self.efficiencystore[i],
                            efficiency_dispatch=self.efficiencydispatch[i],
                            standing_loss=self.standingloss[i],
                            env_f=self.env_f[i],
                            env_v=self.env_v[i],
                            water_f=self.water_f[i],
                            water_v=self.water_v[i],
                            )

            elif self.kind[i] == "energy":
                network.add("Bus", "existing battery bus " + i,
                            carrier='electricity',
                            x=self.x[i],
                            y=self.y[i]
                            )
                network.add("Link",
                            "from existing battery link " + i,
                            bus0="electricity bus " + i,
                            bus1="existing battery bus " + i,
                            p_nom=self.capacity[i] / self.efficiencydispatch[i],
                            efficiency=self.efficiencydispatch[i],
                            marginal_cost=functions.calculate_marginal_costs(self.fuelcost[i], self.variableom[i],
                                                                             self.efficiencydispatch[i]),
                            # marginal cost of the production of 1MWh
                            )
                network.add("Link",
                            "to existing battery link " + i,
                            bus0="existing battery bus " + i,
                            bus1="electricity bus " + i,
                            p_nom=self.capacity[i],
                            efficiency=self.efficiencystore[i],
                            )
                network.add("Store",  # PyPSA component
                            "existing battery " + i,  # Name of the element
                            bus="existing battery bus " + i,  # Name of the bus to which the storage is attached
                            e_nom=self.capacity[i],  # Nominal power (MW)
                            e_cyclic=True,
                            standing_loss=self.standingloss[i],
                            e_min_pu=self.eminpu[i],
                            e_max_pu=self.emaxpu[i],
                            env_f=self.env_f[i],
                            env_v=self.env_v[i],
                            water_f=self.water_f[i],
                            water_v=self.water_v[i],
                            )

    def constraints_existing_battery(self, n, model, horizon):
        # TODO conserver le n en paramètre (Pyomo)
        snap = pd.Series(horizon).to_xarray()
        snap = snap.rename({'index': 'snapshots'})

        def soc_batterie_1(m, t):
            """
            Constraint to bound state of charge of already installed storages (as StorageUnit)
            :param m: model
            :param t: snapshot
            :return:
            """
            return m.variables["StorageUnit-state_of_charge"][t, "existing battery " + i] >= self.capacity[i] * \
                self.eminpu[i]

        def soc_batterie_2(m, t):
            """
            Constraint to bound state of charge of already installed storages (as StorageUnit)
            :param m: model
            :param t: snapshot
            :return:
            """
            return m.variables["StorageUnit-state_of_charge"][t, "existing battery " + i] <= self.capacity[i] * \
                self.emaxpu[i]

        for i in self.places:
            if self.kind[i] == "power":
                model.add_constraints(soc_batterie_1, coords=(snap,), name="soc_batterie_1_" + str(i))
                model.add_constraints(soc_batterie_2, coords=(snap,), name="soc_batterie_2_" + str(i))
            elif self.kind[i] == "energy":
                # Base constraint for Store component as StorageUnit component
                model.add_constraints(model.variables["Store-e_nom"]["existing battery " + i] -
                                      model.variables["Link-p_nom"]["to existing battery link " + i]
                                      * self.efficiencystore[i] == 0,
                                      name="store_fix_1_" + str(i))
                model.add_constraints(model.variables["Store-e_nom"]["existing battery " + i] -
                                      model.variables["Link-p_nom"]["from existing battery link " + i] *
                                      self.efficiencydispatch[i] == 0,
                                      name="store_fix_2_" + str(i))


class AdditionalStorages:
    def __init__(self, data, ps):
        self.data = data.loc[data["technology"] == "electrical"]
        if self.data["place"].iloc[0] == "all":
            self.places = ps.index
        else:
            self.places = self.data["place"]

    def import_storages(self, network, ps):
        network.madd("Bus",
                     "additional battery bus " + self.places,
                     carrier='electricity',
                     x=ps.loc[ps.index.isin(self.places.tolist())]["Long"],
                     y=ps.loc[ps.index.isin(self.places.tolist())]["Lat"]
                     )
        network.madd("Link",
                     "from additional battery link " + self.places,
                     bus0=("electricity bus " + self.places).tolist(),
                     bus1=("additional battery bus " + self.places).tolist(),
                     p_nom=0,
                     p_nom_extendable=True,
                     efficiency=self.data["efficiency dispatch"].iloc[0],
                     marginal_cost=functions.calculate_marginal_costs(self.data["fuel_cost"].iloc[0],
                                                                      self.data["variable_OM"].iloc[0],
                                                                      self.data["efficiency dispatch"].iloc[0]),
                     # marginal cost of the production of 1MWh
                     )
        network.madd("Link",
                     "to additional battery link " + self.places,
                     bus1=("electricity bus " + self.places).tolist(),
                     bus0=("additional battery bus " + self.places).tolist(),
                     p_nom=0,
                     p_nom_extendable=True,
                     efficiency=self.data["efficiency store"].iloc[0],
                     marginal_cost=functions.calculate_marginal_costs(self.data["fuel_cost"].iloc[0],
                                                                      self.data["variable_OM"].iloc[0],
                                                                      self.data["efficiency store"].iloc[0]),
                     # marginal cost of the production of 1MWh
                     )

        network.madd("Store",  # PyPSA component
                     "additional battery " + self.places,  # Name of the element
                     bus=("additional battery bus " + self.places).tolist(),
                     # Name of the bus to which the storage is attached
                     e_nom=0,  # Nominal power (MW)
                     e_nom_extendable=True,  # The capacity can be extended
                     e_nom_min=0,  # Minimum value of capacity
                     # e_nom_max=25,
                     e_cyclic=True,
                     standing_loss=self.data["standing loss"].iloc[0],
                     e_min_pu=self.data["soc min"].iloc[0],
                     e_max_pu=self.data["soc max"].iloc[0],
                     capital_cost=functions.calculate_capital_costs(self.data["discount_rate"].iloc[0],
                                                                    self.data["lifetime"].iloc[0],
                                                                    self.data["fixed_OM (%)"].iloc[0],
                                                                    self.data["fixed_OM (tot)"].iloc[0],
                                                                    self.data["CAPEX"].iloc[0],
                                                                    1),  # €/MWh, cost of extending e_nom by 1 MWh
                     marginal_cost=functions.calculate_marginal_costs(self.data["fuel_cost"].iloc[0],
                                                                      self.data["variable_OM"].iloc[0],
                                                                      self.data["efficiency store"].iloc[0] +
                                                                      self.data["efficiency dispatch"].iloc[0]),
                     # marginal cost of the production of 1MWh
                     env_f=self.data["env_f"].iloc[0],
                     env_v=self.data["env_v"].iloc[0],
                     water_f=self.data["water_f"].iloc[0],
                     water_v=self.data["water_v"].iloc[0],
                     )

    def constraints_additionnal_battery(self, n, model):
        coords = pd.Series(self.places.to_list()).to_xarray()
        coords = coords.rename({'index': 'bus'})

        def link_to_battery(m, k):
            """
            Base constraint for Store component as StorageUnit component
            :param m: model
            :param k: substation
            :return:
            """
            return m.variables["Store-e_nom"]["additional battery " + k] - m.variables["Link-p_nom"][
                "to additional battery link " + k] * self.data["efficiency store"].iloc[0] == 0

        def link_from_battery(m, k):
            """
            Base constraint for Store component as StorageUnit component
            :param m: model
            :param k: substation
            :return:
            """
            return m.variables["Store-e_nom"]["additional battery " + k] - m.variables["Link-p_nom"][
                "from additional battery link " + k] * self.data["efficiency dispatch"].iloc[0] == 0

        model.add_constraints(link_to_battery, coords=(coords,), name="store_fix_1")
        model.add_constraints(link_from_battery, coords=(coords,), name="store_fix_2")
