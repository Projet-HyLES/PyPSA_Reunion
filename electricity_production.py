import functions_used as functions
import pandas as pd


# Definition of the electricity generation technologies : PV, wind, base

class PV:
    def __init__(self, data):
        self.gamma = 0.0599782  # Mounting type of the system parameter (°C.m²/W)
        self.alpha = -0.0035  # Temperature coefficient (%/°C)
        self.carrier = data.loc[data["technology"] == "PV"].squeeze()["carrier"]
        self.efficiency = data.loc[data["technology"] == "PV"].squeeze()["efficiency"]
        self.fuelcost = data.loc[data["technology"] == "PV"].squeeze()["fuel_cost"]
        self.variableom = data.loc[data["technology"] == "PV"].squeeze()["variable_OM"]
        self.discountrate = data.loc[data["technology"] == "PV"].squeeze()["discount_rate"]
        self.lifetime = data.loc[data["technology"] == "PV"].squeeze()["lifetime"]
        self.fixedOM_p = data.loc[data["technology"] == "PV"].squeeze()["fixed_OM (%)"]
        self.fixedOM_t = data.loc[data["technology"] == "PV"].squeeze()["fixed_OM (tot)"]
        self.CAPEX = data.loc[data["technology"] == "PV"].squeeze()["nominal investment"]
        self.env_f = data.loc[data["technology"] == "PV"].squeeze()["env_f"]
        self.env_v = data.loc[data["technology"] == "PV"].squeeze()["env_v"]
        self.water_f = data.loc[data["technology"] == "PV"].squeeze()["water_f"]
        self.water_v = data.loc[data["technology"] == "PV"].squeeze()["water_v"]

    def import_pv(self, network, tot, t, r, ps, ext):
        Tm = t + self.gamma * r
        # We make the model take as much electricity from PV sources as possible
        if ext:
            current_data = pd.read_csv(
                network.data_path + '/registre-des-installations-de-production-et-de-stockage.csv',
                encoding='latin-1', sep=';',
                usecols=['Poste source', 'Filière', 'Puissance installée (kW)'])
            current_data = current_data[current_data["Filière"] == "PV"]
            network.add("Generator",  # PyPSA component
                        ps + " PV",  # Name of the element
                        bus="electricity bus " + ps,
                        # Name of the bus to which the technology is attached
                        carrier=self.carrier,  # Name of the carrier of the technology
                        p_nom_extendable=True,
                        p_nom=round(current_data[current_data["Poste source"] == ps]['Puissance installée (kW)'].sum(), 2) / 1000,
                        # Nominal power (MW)
                        p_nom_min=round(current_data[current_data["Poste source"] == ps][
                                      'Puissance installée (kW)'].sum(), 2) / 1000,
                        p_nom_max=tot,
                        p_min_pu=(r / 1000) * (1 + self.alpha * (Tm - 25)) * self.efficiency,  # Minimum output
                        p_max_pu=(r / 1000) * (1 + self.alpha * (Tm - 25)) * self.efficiency,  # Maximum output
                        marginal_cost=functions.calculate_marginal_costs(self.fuelcost, self.variableom,
                                                                         self.efficiency),
                        # Marginal cost of production of 1MWh
                        capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                       self.fixedOM_t, self.CAPEX, 1),
                        env_f=self.env_f,
                        env_v=self.env_v,
                        water_f=self.water_f,
                        water_v=self.water_v,
                        )
        else:
            network.add("Generator",  # PyPSA component
                        ps + " PV",  # Name of the element
                        bus="electricity bus " + ps,
                        # Name of the bus to which the technology is attached
                        carrier=self.carrier,  # Name of the carrier of the technology
                        p_nom=tot,  # Nominal power (MW)
                        p_min_pu=(r / 1000) * (1 + self.alpha * (Tm - 25)) * self.efficiency,  # Minimum output
                        p_max_pu=(r / 1000) * (1 + self.alpha * (Tm - 25)) * self.efficiency,  # Maximum output
                        marginal_cost=functions.calculate_marginal_costs(self.fuelcost, self.variableom,
                                                                         self.efficiency),
                        # Marginal cost of production of 1MWh
                        env_f=self.env_f,
                        env_v=self.env_v,
                        water_f=self.water_f,
                        water_v=self.water_v,
                        )


class Wind:
    def __init__(self, data, brand):
        self.rho = 1.225  # Air density (kg/m3)
        # TODO changer la distinction ? Plus de précision ? Plus de choix de marque ?
        if brand == "Vestas":  # onshore
            self.v_min = 3  # Minimal wind speed (m/s)
            self.v_nom = 11.5  # Nominal wind speed (m/s)
            self.v_max = 20  # Maximal wind speed (m/s)
            self.diam = 110  # Rotor swept area exposed to the wind (m)
            self.filiere = "Eolien"
        elif brand == "Haliade":  # offshore TODO à update
            # self.v_min = 3  # Minimal wind speed (m/s)
            # self.v_nom = 12  # Nominal wind speed (m/s)
            # self.v_max = 25  # Maximal wind speed (m/s)
            # self.diam = 150.95  # Rotor swept area exposed to the wind (m)
            self.v_min = 3  # Minimal wind speed (m/s)
            self.v_nom = 11.5  # Nominal wind speed (m/s)
            self.v_max = 20  # Maximal wind speed (m/s)
            self.diam = 110  # Rotor swept area exposed to the wind (m)
            self.filiere = "Eolien offshore"
        self.carrier = data.loc[data["technology"] == self.filiere].squeeze()["carrier"]
        self.efficiency = data.loc[data["technology"] == self.filiere].squeeze()["efficiency"]
        self.fuelcost = data.loc[data["technology"] == self.filiere].squeeze()["fuel_cost"]
        self.variableom = data.loc[data["technology"] == self.filiere].squeeze()["variable_OM"]
        self.discountrate = data.loc[data["technology"] == self.filiere].squeeze()["discount_rate"]
        self.lifetime = data.loc[data["technology"] == self.filiere].squeeze()["lifetime"]
        self.fixedOM_p = data.loc[data["technology"] == self.filiere].squeeze()["fixed_OM (%)"]
        self.fixedOM_t = data.loc[data["technology"] == self.filiere].squeeze()["fixed_OM (tot)"]
        self.CAPEX = data.loc[data["technology"] == self.filiere].squeeze()["nominal investment"]
        self.env_f = data.loc[data["technology"] == self.filiere].squeeze()["env_f"]
        self.env_v = data.loc[data["technology"] == self.filiere].squeeze()["env_v"]
        self.water_f = data.loc[data["technology"] == self.filiere].squeeze()["water_f"]
        self.water_v = data.loc[data["technology"] == self.filiere].squeeze()["water_v"]

    def import_wind(self, network, tot, v, ps, ext):
        # TODO modèle de production eolien offshore à changer
        # We make the model take as much electricity from wind sources as possible
        power = pd.DataFrame(index=network.horizon, columns=["Power"])
        for i in network.horizon:
            power["Power"].loc[i] = functions.prod_vestas(self.v_min, self.v_max, self.v_nom, self.rho, self.diam,
                                                          v.loc[i], tot, tot / 2)
        if ext:
            current_data = pd.read_csv(
                network.data_path + '/registre-des-installations-de-production-et-de-stockage.csv',
                encoding='latin-1', sep=';',
                usecols=['Poste source', 'Filière', 'Puissance installée (kW)'])
            current_data = current_data[current_data["Filière"] == self.filiere]

            network.add("Generator",  # PyPSA component
                        ps + " " + self.filiere,  # Name of the element
                        bus="electricity bus " + ps,
                        # Name of the bus to which the technology is attached
                        carrier=self.carrier,  # Name of the carrier of the technology
                        p_nom_extendable=True,
                        p_nom=round(current_data[current_data["Poste source"] == ps]['Puissance installée (kW)'].sum(), 2) / 1000,
                        # Nominal power (MW)
                        p_nom_min=round(current_data[current_data["Poste source"] == ps]['Puissance installée (kW)'].sum(), 2) / 1000,
                        p_nom_max=tot,
                        p_min_pu=power["Power"] / (tot * 1000),  # Minimum output
                        p_max_pu=power["Power"] / (tot * 1000),  # Maximum output
                        marginal_cost=functions.calculate_marginal_costs(self.fuelcost, self.variableom,
                                                                         self.efficiency),
                        capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                       self.fixedOM_t, self.CAPEX, 1),
                        env_f=self.env_f,
                        env_v=self.env_v,
                        water_f=self.water_f,
                        water_v=self.water_v,
                        )
        else:
            network.add("Generator",  # PyPSA component
                        ps + " " + self.filiere,  # Name of the element
                        bus="electricity bus " + ps,
                        # Name of the bus to which the technology is attached
                        carrier=self.carrier,  # Name of the carrier of the technology
                        p_nom=tot,  # Nominal power (MW)
                        p_min_pu=power["Power"] / (tot * 1000),  # Minimum output
                        p_max_pu=power["Power"] / (tot * 1000),  # Maximum output
                        marginal_cost=functions.calculate_marginal_costs(self.fuelcost,
                                                                         self.variableom,
                                                                         self.efficiency),
                        capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                       self.fixedOM_t, self.CAPEX, 1),
                        env_f=self.env_f,
                        env_v=self.env_v,
                        water_f=self.water_f,
                        water_v=self.water_v,
                        )


class BaseProduction:
    def __init__(self, data, filiere):
        self.filiere = filiere
        self.carrier = data.loc[data["technology"] == self.filiere].squeeze()["carrier"]
        self.efficiency = data.loc[data["technology"] == self.filiere].squeeze()["efficiency"]
        self.fuelcost = data.loc[data["technology"] == self.filiere].squeeze()["fuel_cost"]
        self.variableom = data.loc[data["technology"] == self.filiere].squeeze()["variable_OM"]
        self.discountrate = data.loc[data["technology"] == self.filiere].squeeze()["discount_rate"]
        self.lifetime = data.loc[data["technology"] == self.filiere].squeeze()["lifetime"]
        self.fixedOM_p = data.loc[data["technology"] == self.filiere].squeeze()["fixed_OM (%)"]
        self.fixedOM_t = data.loc[data["technology"] == self.filiere].squeeze()["fixed_OM (tot)"]
        self.CAPEX = data.loc[data["technology"] == self.filiere].squeeze()["nominal investment"]
        self.pminpu = data.loc[data["technology"] == self.filiere].squeeze()["p_min_pu"]
        self.pmaxpu = data.loc[data["technology"] == self.filiere].squeeze()["p_max_pu"]
        self.cf = data.loc[data["technology"] == self.filiere].squeeze()["capacity factor"]
        self.max_y = data.loc[data["technology"] == self.filiere].squeeze()["max_year"]
        self.min_y = data.loc[data["technology"] == self.filiere].squeeze()["min_year"]
        self.env_f = data.loc[data["technology"] == self.filiere].squeeze()["env_f"]
        self.env_v = data.loc[data["technology"] == self.filiere].squeeze()["env_v"]
        self.water_f = data.loc[data["technology"] == self.filiere].squeeze()["water_f"]
        self.water_v = data.loc[data["technology"] == self.filiere].squeeze()["water_v"]

    def import_base(self, network, tot, ps, ext):
        if ext:
            current_data = pd.read_csv(
                network.data_path + '/registre-des-installations-de-production-et-de-stockage.csv',
                encoding='latin-1', sep=';',
                usecols=['Poste source', 'Filière', 'Puissance installée (kW)'])
            current_data = current_data[current_data["Filière"] == self.filiere]

            network.add("Generator",  # PyPSA component
                        ps + " " + self.filiere,  # Name of the element
                        bus="electricity bus " + ps,
                        # Name of the bus to which the technology is attached
                        carrier=self.carrier,  # Name of the carrier of the technology
                        p_nom_extendable=True,
                        p_nom=round(current_data[current_data["Poste source"] == ps]['Puissance installée (kW)'].sum(), 2) / 1000,
                        # Nominal power (MW)
                        p_nom_min=round(current_data[current_data["Poste source"] == ps]['Puissance installée (kW)'].sum(), 2) / 1000,
                        p_nom_max=tot,
                        p_min_pu=self.pminpu,  # Minimum output
                        p_max_pu=self.pmaxpu,  # Maximum output
                        marginal_cost=functions.calculate_marginal_costs(self.fuelcost, self.variableom,
                                                                         self.efficiency),
                        capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                       self.fixedOM_t, self.CAPEX, 1),
                        env_f=self.env_f,
                        env_v=self.env_v,
                        water_f=self.water_f,
                        water_v=self.water_v,
                        )
        else:
            network.add("Generator",  # PyPSA component
                        ps + " " + self.filiere,  # Name of the element
                        bus="electricity bus " + ps,
                        # Name of the bus to which the technology is attached
                        carrier=self.carrier,  # Name of the carrier of the technology
                        p_nom=tot,  # Nominal power (MW)
                        p_min_pu=self.pminpu,  # Minimum output, fixed with comparison to data given
                        p_max_pu=self.pmaxpu,  # Maximum output, fixed with comparison to data given
                        efficiency=self.efficiency,
                        # Ratio between primary energy and electrical energy
                        marginal_cost=functions.calculate_marginal_costs(self.fuelcost,
                                                                         self.variableom,
                                                                         self.efficiency),
                        capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                       self.fixedOM_t, self.CAPEX, 1),
                        env_f=self.env_f,
                        env_v=self.env_v,
                        water_f=self.water_f,
                        water_v=self.water_v,
                        # committable=row_param['committable'],  # Use unit commitment
                        # start_up_cost=row_param['start up cost'] * total_capa,  # Cost to start up the technology (€/MW)
                        # shut_down_cost=row_param['shut down cost'],  # Cost to shut down the technology
                        # min_up_time=row_param['min up time'],  # Minimum up time required
                        # min_down_time=row_param['min down time'],  # Minimum down time required
                        # up_time_before=row_param['up time bf'],  # Time up before the modeling
                        # down_time_before=row_param['down time bf'],  # Time down before the modeling
                        # ramp_limit_up=row_param['ramp limit up'],  # Maximum active power increase
                        # ramp_limit_down=row_param['ramp limit down'],  # Maximum active power decrease
                        # ramp_limit_start_up=row_param['ramp limit on'],  # Maximum active power increase at start up
                        # ramp_limit_shut_down=row_param['ramp limit off'],  # Maximum active power decrease at start up
                        )

    def constraint_disp(self, n, model, snap, xa, ext):
        """Constraint to define the disponibility of power generation facilities"""

        def disp(m, k):
            if ext:
                return sum(m.variables["Generator-p"][j, k] for j in list(snap)) - snap.size * self.cf * m.variables["Generator-p_nom"][k] <= 0
            else:
                return sum(m.variables["Generator-p"][j, k] for j in list(snap)) <= snap.size * self.cf * n.generators.p_nom[k]

        model.add_constraints(disp, coords=(xa,), name="disp_" + self.filiere)

    def constraint_min_max(self, n, model, snap, xa, ext):
        """
        Constraint to define the minimum and maximum yearly generation of power generation facilities
        :param n: network
        :param model: model
        :param snap: snapshots
        :param xa: xarray of the technologies
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        :return:
        """
        if ext:
            model.add_constraints(
                sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) -
                self.min_y * sum(model.variables["Generator-p_nom"][i] for i in xa) / sum(
                    n.generators["p_nom_max"][i] for i in xa) >= 0,
                name="limit1_" + str(self.filiere))
            model.add_constraints(
                sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) -
                self.max_y * sum(model.variables["Generator-p_nom"][i] for i in xa) / sum(
                    n.generators["p_nom_max"][i] for i in xa) <= 0,
                name="limit2_" + str(self.filiere))
        else:
            model.add_constraints(
                sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) >= self.min_y,
                name="limit1_" + str(self.filiere))
            model.add_constraints(
                sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) <= self.max_y,
                name="limit2_" + str(self.filiere))
