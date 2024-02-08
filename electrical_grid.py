import re
import pandas as pd
import functions_used as functions


# Definition of the electrical grid : substations, electrical lines, existing batteries and additional batteries.


class ElectricalGrid:
    """
    Represents the electrical grid in the network.
    """
    R = 0.06  # Reactance, source : cours de Robin
    X = 0.37  # Resistance
    # Power line conductors capacity and cost (€/km), source : Vers l'autonomie energetique des ZNI - ADEME
    COST_COND = 212  # TODO 'a' de la linéarité cout/MVA (à update plus tard)
    CAP_MAX_1 = 39
    COST_COND_1 = 4200
    CAP_MAX_2 = 50
    COST_COND_2 = 7400
    CAP_MAX_3 = 67
    COST_COND_3 = 10400
    CAP_MAX_4 = 88
    COST_COND_4 = 14900
    COST_FIX = 25000  # €/km, cost for extending s_nom by 1 MVA (source rapport ADEME)
    ISLAND_FACTOR = 1.2  # Islanding factor
    ENV_F = 0.01  # Environmental impact fixed
    ENV_V = 0.01  # Environmental impact variable
    WATER_F = 1  # Water factor
    WATER_V = 1  # Water voltage

    def __init__(self, network):
        """
        Initialize the ElectricalGrid object.

        :param network: The network object.
        """
        self.network = network
        self.lines = self.network.eleclines
        self.buses_name = self.network.data["postes"].index
        self.buses_vnom = self.network.data["postes"]["Voltage"].tolist()
        self.buses_x = self.network.data["postes"]["Long"].tolist()
        self.buses_y = self.network.data["postes"]["Lat"].tolist()

    def import_buses(self):
        """
        Import buses into the network.
        """
        self.network.madd(
            "Bus",
            "electricity bus " + self.buses_name,
            v_nom=self.buses_vnom,
            x=self.buses_x, # Longitude
            y=self.buses_y, # Latitude
        )

    def import_lines(self):
        """
        Import lines into the network.
        """
        pattern = r'[0-9]'
        for _, row in self.lines.iterrows():
            if row.eq("Piquage").any():
                # The line can be ignored, the substation is linked to the other substations with a line "well named"
                print("INFO: line {} is incomplete and joins another one.".format(row["Nom de la ligne"]))
                continue

            if len(row["Nom de la ligne"].split('/')) > 2:
                # One line can link more than two substations
                for i in range(len(row["Nom de la ligne"].split('/')) - 1):
                    for j in range(i + 1, len(row["Nom de la ligne"].split('/'))):
                        new_name = [row["bus" + str(i)], row["bus" + str(j)],
                                    re.sub(pattern, '', row["bus" + str(i)]), re.sub(pattern, '', row["bus" + str(j)])]
                        self.importing_line(row, new_name, row["Capacite (MVA)"], self.CAP_MAX_4,
                                            self.COST_COND)
            else:
                new_name = [row["bus0"], row["bus1"],
                            re.sub(pattern, '', row["bus0"]), re.sub(pattern, '', row["bus1"])]
                self.importing_line(row, new_name, row["Capacite (MVA)"], self.CAP_MAX_4, self.COST_COND)

    def importing_line(self, row, name, cap, cap_max, cost):
        """
        Importing a power line with PyPSA structure.
        """
        self.network.add(
            "Line",
            name[0] + " to " + name[1] + " " + row["Type"],
            bus0="electricity bus " + name[2],
            bus1="electricity bus " + name[3],
            s_nom=cap,
            s_nom_extendable=True,
            s_nom_min=cap,
            s_nom_max=cap_max,
            x=self.X,
            r=self.R,
            capital_cost=(self.COST_FIX + cost) * row["Longueur (km)"] * self.ISLAND_FACTOR,  # TODO Capital cost of extending s_nom by 1 MVA.
            env_f=self.ENV_F
        )


class ExistingStorages:
    """
    Represents the existing storages in the network.
    """

    def __init__(self, network):
        """
        Initialize the ExistingStorages object.

        :param network: The network object.
        """
        self.network = network
        self.batteries = self.network.data["batteries"]
        self.places = self.batteries.index
        self.extract_attributes()

    def extract_attributes(self):
        """
        Extract attributes from the battery data and assign them to instance variables.
        """
        attributes_mapping = {
            "kind": "kind",
            "carrier": "carrier",
            "fuelcost": "fuel_cost",
            "variableom": "variable_OM",
            "efficiencystore": "efficiency store",
            "efficiencydispatch": "efficiency dispatch",
            "standingloss": "standing loss",
            "capacity": "capacity",
            "power": "power",
            "max_hours": "max hours",
            "eminpu": "soc min",
            "emaxpu": "soc max",
            "env_f": "env_f",
            "env_v": "env_v",
            "water_f": "water_f",
            "water_v": "water_v"
        }

        for attr, column_name in attributes_mapping.items():
            setattr(self, attr, self.batteries[column_name])

    def import_storages(self):
        """
        Import existing storages into the network.
        """
        for i in self.places:
            if self.kind[i] == "power":
                self.import_power_storage(i)
            elif self.kind[i] == "energy":
                self.import_energy_storage(i)

    def import_power_storage(self, i):
        """
        Import an existing power storage into the network.

        :param i: The index of the battery.
        :type i: int
        """
        self.network.add(
            "StorageUnit",
            "existing battery " + i,
            bus="electricity bus " + i,
            carrier=self.carrier[i],
            p_nom=self.power[i],
            max_hours=self.max_hours[i],
            marginal_cost=self.calculate_marginal_costs(i),
            cyclic_state_of_charge=True,
            efficiency_store=self.efficiencystore[i],
            efficiency_dispatch=self.efficiencydispatch[i],
            standing_loss=self.standingloss[i],
            env_f=self.env_f[i],
            env_v=self.env_v[i],
            water_f=self.water_f[i],
            water_v=self.water_v[i]
        )

    def import_energy_storage(self, i):
        """
        Import an existing energy storage into the network.

        :param i: The index of the battery.
        :type i: int
        """
        self.network.add(
            "Bus",
            "existing battery bus " + i,
            carrier="electricity",
            x=self.network.data["postes"].loc[self.network.data["postes"].index == i]["Long"],
            y=self.network.data["postes"].loc[self.network.data["postes"].index == i]["Lat"]
        )

        self.network.add(
            "Link",
            "from existing battery link " + i,
            bus0="electricity bus " + i,
            bus1="existing battery bus " + i,
            p_nom=self.power[i],
            efficiency=self.efficiencydispatch[i],
            marginal_cost=self.calculate_marginal_costs(i)
        )

        self.network.add(
            "Link",
            "to existing battery link " + i,
            bus0="existing battery bus " + i,
            bus1="electricity bus " + i,
            p_nom=self.power[i],
            efficiency=self.efficiencystore[i]
        )

        self.network.add(
            "Store",
            "existing battery " + i,
            bus="existing battery bus " + i,
            carrier=self.carrier[i],
            e_nom=self.capacity[i],
            e_cyclic=True,
            standing_loss=self.standingloss[i],
            e_min_pu=self.eminpu[i],
            e_max_pu=self.emaxpu[i],
            env_f=self.env_f[i],
            env_v=self.env_v[i],
            water_f=self.water_f[i],
            water_v=self.water_v[i]
        )

    def calculate_marginal_costs(self, i):
        """
        Calculate the marginal costs of the battery of the production of 1MWh.

        :param i: The index of the battery.
        :type i: int

        :return: The calculated marginal costs.
        :rtype: float
        """
        return functions.calculate_marginal_costs(
            self.fuelcost[i], self.variableom[i], self.efficiencydispatch[i]
        )

    def constraints_existing_battery(self, n, model, horizon):
        """
        Add constraints for the existing batteries to the model.

        :param n: The number of snapshots.
        :type n: int

        :param model: The optimization model.
        :type model: <type of model>

        :param horizon: The horizon for which the model is built.
        :type horizon: pandas.DatetimeIndex
        """
        # TODO conserver le n en paramètre (Pyomo)
        snap = pd.Series(horizon).to_xarray()
        snap = snap.rename({"index": "snapshots"})

        for place in self.places:
            if self.kind[place] == "power":
                self.add_power_constraints(model, place, snap)
            # elif self.kind[place] == "energy":
            #     self.add_energy_constraints(model, place)

    def add_power_constraints(self, model, place, snap):
        """
        Add power constraints for an existing power storage to the model.

        :param model: The optimization model.
        :type model: <type of model>

        :param place: The place where the battery is located.
        :type place: int

        :param snap: The snapshots.
        :type snap: pandas.Series
        """
        def soc_batterie_1(m, t):
            """
            Constraint to bound state of charge of already installed storages (as StorageUnit).
            :param m: model.
            :param t: snapshot.
            :return: The constraint expression.
            """
            return m.variables["StorageUnit-state_of_charge"][t, "existing battery " + place] >= self.power[place] * self.eminpu[place]

        def soc_batterie_2(m, t):
            """
            Constraint to bound state of charge of already installed storages (as StorageUnit).
            :param m: model.
            :param t: snapshot.
            :return: The constraint expression.
            """
            return m.variables["StorageUnit-state_of_charge"][t, "existing battery " + place] <= self.power[place] * self.emaxpu[place]

        model.add_constraints(soc_batterie_1, coords=(snap,), name="soc_batterie_1_" + str(place))
        model.add_constraints(soc_batterie_2, coords=(snap,), name="soc_batterie_2_" + str(place))

    # def add_energy_constraints(self, model, place):
    #     """
    #     Base constraint for Store component as StorageUnit component.
    #
    #     :param model: The optimization model.
    #     :type model: <type of model>
    #
    #     :param place: The place where the battery is located.
    #     :type place: int
    #     """
    #     model.add_constraints(
    #         model.variables["Store-e_nom"]["existing battery " + place] -
    #         model.variables["Link-p_nom"]["to existing battery link " + place] * self.efficiencystore[place] == 0,
    #         name="store_fix_1_" + str(place)
    #     )
    #
    #     model.add_constraints(
    #         model.variables["Store-e_nom"]["existing battery " + place] -
    #         model.variables["Link-p_nom"]["from existing battery link " + place] * self.efficiencydispatch[place] == 0,
    #         name="store_fix_2_" + str(place)
    #     )
    # No need as power and capacity are fixed and non extendable


class AdditionalStorages:
    """
    Imports additional storages into the network.
    """

    def __init__(self, network):
        """
        Initialize the AdditionalStorages object.

        :param network: The network object.
        """
        self.network = network
        self.dataStorage = self.network.data["storage"].loc[self.network.data["storage"]["technology"] == "electrical"]

        if self.dataStorage["place"].iloc[0] == "all":
            self.places = self.network.data["postes"].index
        else:
            self.places = self.dataStorage["place"]

    def import_storages(self):
        """
        Import the additional storages into the network.
        """
        ps = self.network.data["postes"]

        additional_battery_bus = "additional battery bus " + self.places
        electricity_bus = "electricity bus " + self.places

        self.network.madd(
            "Bus",
            additional_battery_bus,
            carrier='electricity',
            x=ps.loc[ps.index.isin(self.places)]["Long"],
            y=ps.loc[ps.index.isin(self.places)]["Lat"]
        )

        self.network.madd(
            "Link",
            "from additional battery link " + self.places,
            bus0=electricity_bus,
            bus1=additional_battery_bus,
            p_nom=0,
            p_nom_extendable=True,
            efficiency=self.dataStorage["efficiency dispatch"].iloc[0],
            marginal_cost=functions.calculate_marginal_costs(
                self.dataStorage["fuel_cost"].iloc[0],
                self.dataStorage["variable_OM"].iloc[0],
                self.dataStorage["efficiency dispatch"].iloc[0]
            ),
        )

        self.network.madd(
            "Link",
            "to additional battery link " + self.places,
            bus1=electricity_bus,
            bus0=additional_battery_bus,
            p_nom=0,
            p_nom_extendable=True,
            efficiency=self.dataStorage["efficiency store"].iloc[0],
        )

        self.network.madd(
            "Store",
            "additional battery " + self.places,
            bus=additional_battery_bus,
            e_nom=0, # Nominal power (MW)
            e_nom_extendable=True,
            e_nom_min=0,
            # e_nom_max=25,
            e_cyclic=True,
            standing_loss=self.dataStorage["standing loss"].iloc[0],
            e_min_pu=self.dataStorage["soc min"].iloc[0],
            e_max_pu=self.dataStorage["soc max"].iloc[0],
            # €/MWh, cost of extending e_nom by 1 MWh
            capital_cost=functions.calculate_capital_costs(
                self.dataStorage["discount_rate"].iloc[0],
                self.dataStorage["lifetime"].iloc[0],
                self.dataStorage["fixed_OM (%)"].iloc[0],
                self.dataStorage["fixed_OM (tot)"].iloc[0],
                self.dataStorage["CAPEX"].iloc[0],
                1
            ),
            env_f=self.dataStorage["env_f"].iloc[0],
            env_v=self.dataStorage["env_v"].iloc[0],
            water_f=self.dataStorage["water_f"].iloc[0],
            water_v=self.dataStorage["water_v"].iloc[0],
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
                "to additional battery link " + k] * self.dataStorage["efficiency store"].iloc[0] == 0

        def link_from_battery(m, k):
            """
            Base constraint for Store component as StorageUnit component
            :param m: model
            :param k: substation
            :return:
            """
            return m.variables["Store-e_nom"]["additional battery " + k] - m.variables["Link-p_nom"][
                "from additional battery link " + k] * self.dataStorage["efficiency dispatch"].iloc[0] == 0

        model.add_constraints(link_to_battery, coords=(coords,), name="store_fix_1")
        model.add_constraints(link_from_battery, coords=(coords,), name="store_fix_2")
