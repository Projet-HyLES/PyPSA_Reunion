import functions_used as functions
import pandas as pd
import numpy as np
from windpowerlib import ModelChain, WindTurbine
import fnmatch
import os

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
        self.CAPEX = data.loc[data["technology"] == "PV"].squeeze()["nominal investment variable"]
        self.env_f = data.loc[data["technology"] == "PV"].squeeze()["env_f"]
        self.env_v = data.loc[data["technology"] == "PV"].squeeze()["env_v"]
        self.water_f = data.loc[data["technology"] == "PV"].squeeze()["water_f"]
        self.water_v = data.loc[data["technology"] == "PV"].squeeze()["water_v"]

    def import_pv(self, network, tot, t, r, ps, ext):
        Tm = t + self.gamma * r
        # We make the model take as much electricity from PV sources as possible
        if ext:
            current_data = pd.read_csv(
                network.data_dir + '/registre-des-installations-de-production-et-de-stockage.csv',
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
                        capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                       self.fixedOM_t, self.CAPEX, 1),
                        # Marginal cost of production of 1MWh
                        env_f=self.env_f,
                        env_v=self.env_v,
                        water_f=self.water_f,
                        water_v=self.water_v,
                        )


class ETM:
    def __init__(self, data):
        self.carrier = data.loc[data["technology"] == "ETM"].squeeze()["carrier"]
        self.efficiency = data.loc[data["technology"] == "ETM"].squeeze()["efficiency"]
        self.fuelcost = data.loc[data["technology"] == "ETM"].squeeze()["fuel_cost"]
        self.variableom = data.loc[data["technology"] == "ETM"].squeeze()["variable_OM"]
        self.discountrate = data.loc[data["technology"] == "ETM"].squeeze()["discount_rate"]
        self.lifetime = data.loc[data["technology"] == "ETM"].squeeze()["lifetime"]
        self.fixedOM_p = data.loc[data["technology"] == "ETM"].squeeze()["fixed_OM (%)"]
        self.fixedOM_t = data.loc[data["technology"] == "ETM"].squeeze()["fixed_OM (tot)"]
        self.CAPEX_v = data.loc[data["technology"] == "ETM"].squeeze()["nominal investment variable"]
        self.CAPEX_f = data.loc[data["technology"] == "ETM"].squeeze()["nominal investment fixed"]
        self.env_f = data.loc[data["technology"] == "ETM"].squeeze()["env_f"]
        self.env_v = data.loc[data["technology"] == "ETM"].squeeze()["env_v"]
        self.water_f = data.loc[data["technology"] == "ETM"].squeeze()["water_f"]
        self.water_v = data.loc[data["technology"] == "ETM"].squeeze()["water_v"]

    def import_etm(self, network, tot, ps):
        # We make the model take as much electricity from PV sources as possible
        liste_capa = []
        for file in os.listdir(network.data_dir):
            if fnmatch.fnmatch(file, "capacityfactor_etm*"):
                liste_capa.append(0.7*int(file[19:21]))
        etm_file = pd.read_csv(network.data_dir + '/capacityfactor_etm_' + str(round(min(liste_capa, key=lambda x:abs(x-tot))/0.7)) + 'MW.csv', index_col=0)
        etm_file.index = network.horizon
        network.add("Generator",  # PyPSA component
                    ps + " ETM",  # Name of the element
                    bus="electricity bus " + ps,
                    # Name of the bus to which the technology is attached
                    carrier=self.carrier,  # Name of the carrier of the technology
                    p_nom=tot,  # Nominal power (MW)
                    p_min_pu=etm_file['p_net'],  # Minimum output
                    p_max_pu=etm_file['p_net'],  # Maximum output
                    marginal_cost=functions.calculate_marginal_costs(self.fuelcost, self.variableom,
                                                                     self.efficiency),
                    capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                   self.fixedOM_t, self.CAPEX_v, 1),
                    base_CAPEX=functions.calculate_capital_costs(self.discountrate, self.lifetime, self.fixedOM_p,
                                                                 self.fixedOM_t, self.CAPEX_f, 1) - self.fixedOM_t * 1,
                    # Fixed O&M costs must be subtracted, as calculated in functions.calculate_capital_costs
                    env_f=self.env_f,
                    env_v=self.env_v,
                    water_f=self.water_f,
                    water_v=self.water_v,
                    )


class Wind:
    def __init__(self, data, type):
        self.rho = 1.225  # Air density (kg/m3)
        self.type = type

        if self.type == "onshore":  # onshore
            vestas_on = pd.read_csv('/home/afrancoi/PyPSA/Codes/HyLES/Data/Vestas_1.csv', sep=';')
            turbine = {'nominal_power': 2e6,  # in W
                       'hub_height': 80,  # in m
                       'rotor_diameter': 110,
                       'power_curve': pd.DataFrame(
                           data={'value': vestas_on['Puissance (kw)'] * 1000,  # in W
                                 'wind_speed': vestas_on['Vitesse de vent (m/s)']})  # in m/s
                       }
            self.my_turbine = WindTurbine(**turbine)

            self.capa = 2  # Capacity of one turbine (MW)
            self.v_min = 3  # Minimal wind speed (m/s)
            self.v_nom = 11.5  # Nominal wind speed (m/s)
            self.v_max = 20  # Maximal wind speed (m/s)
            self.diam = 110  # Rotor swept area exposed to the wind (m)
            self.filiere = "Eolien"
        elif self.type == "offshore":
            vestas_off = pd.read_csv('/home/afrancoi/PyPSA/Codes/HyLES/Data/Vestas_2.csv', sep=';')
            turbine = {'nominal_power': 9.5e6,  # in W
                       'hub_height': 140,  # in m
                       'rotor_diameter': 164,
                       'power_curve': pd.DataFrame(
                           data={'value': vestas_off['Puissance (kw)'] * 1000,  # in W
                                 'wind_speed': vestas_off['Vitesse de vent (m/s)']})  # in m/s
                       }
            self.my_turbine = WindTurbine(**turbine)

            self.capa = 9.5  # Capacity of one turbine (MW)
            self.v_min = 3.5  # Minimal wind speed (m/s)
            self.v_nom = 14  # Nominal wind speed (m/s)
            self.v_max = 25  # Maximal wind speed (m/s)
            self.diam = 164  # Rotor swept area exposed to the wind (m)
            self.filiere = "Eolien offshore"
        self.carrier = data.loc[data["technology"] == self.filiere].squeeze()["carrier"]
        self.efficiency = data.loc[data["technology"] == self.filiere].squeeze()["efficiency"]
        self.fuelcost = data.loc[data["technology"] == self.filiere].squeeze()["fuel_cost"]
        self.variableom = data.loc[data["technology"] == self.filiere].squeeze()["variable_OM"]
        self.discountrate = data.loc[data["technology"] == self.filiere].squeeze()["discount_rate"]
        self.lifetime = data.loc[data["technology"] == self.filiere].squeeze()["lifetime"]
        self.fixedOM_p = data.loc[data["technology"] == self.filiere].squeeze()["fixed_OM (%)"]
        self.fixedOM_t = data.loc[data["technology"] == self.filiere].squeeze()["fixed_OM (tot)"]
        self.CAPEX = data.loc[data["technology"] == self.filiere].squeeze()["nominal investment variable"]
        self.env_f = data.loc[data["technology"] == self.filiere].squeeze()["env_f"]
        self.env_v = data.loc[data["technology"] == self.filiere].squeeze()["env_v"]
        self.water_f = data.loc[data["technology"] == self.filiere].squeeze()["water_f"]
        self.water_v = data.loc[data["technology"] == self.filiere].squeeze()["water_v"]

    def import_wind(self, network, tot, v, t, ps, ext):
        # We make the model take as much electricity from wind sources as possible
        poweron = pd.DataFrame(index=network.horizon, columns=["Power"])
        for i in network.horizon:
            poweron["Power"].loc[i] = functions.prod_vestas(self.type, self.v_min, self.v_max, self.v_nom, self.rho,
                                                            self.diam,
                                                            v.loc[i], self.capa, tot / self.capa)

        weather = pd.DataFrame(columns=['variable_name', 'pressure', 'temperature', 'wind_speed', 'roughness_length'])
        weather['variable_name'] = network.horizon
        weather['pressure'] = 101325
        weather['temperature'] = np.array(t.tolist()) + 273.15
        weather['wind_speed'] = v.tolist()
        weather['roughness_length'] = 0.15
        weather.columns = pd.MultiIndex.from_tuples([(col, 10) for col in weather.columns])

        mc_my_turbine = ModelChain(self.my_turbine).run_model(weather)
        self.my_turbine.power_output = mc_my_turbine.power_output
        poweroff = self.my_turbine.power_output * tot / self.capa

        if ext:
            current_data = pd.read_csv(
                network.data_dir + '/registre-des-installations-de-production-et-de-stockage.csv',
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
                        p_min_pu=poweron["Power"] / tot,  # Minimum output
                        p_max_pu=poweron["Power"] / tot,  # Maximum output
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
            if self.type == "onshore":  # Modèle Cp fait à la main
                network.add("Generator",  # PyPSA component
                            ps + " " + self.filiere,  # Name of the element
                            bus="electricity bus " + ps,
                            # Name of the bus to which the technology is attached
                            carrier=self.carrier,  # Name of the carrier of the technology
                            p_nom=tot,  # Nominal power (MW)
                            p_min_pu=poweron["Power"] / tot,  # Minimum output
                            p_max_pu=poweron["Power"] / tot,  # Maximum output
                            marginal_cost=functions.calculate_marginal_costs(self.fuelcost,
                                                                             self.variableom,
                                                                             self.efficiency),
                            capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime,
                                                                           self.fixedOM_p,
                                                                           self.fixedOM_t, self.CAPEX, 1),
                            env_f=self.env_f,
                            env_v=self.env_v,
                            water_f=self.water_f,
                            water_v=self.water_v,
                            )
            else:  # Modèle WindPowerLib
                network.add("Generator",  # PyPSA component
                            ps + " " + self.filiere,  # Name of the element
                            bus="electricity bus " + ps,
                            # Name of the bus to which the technology is attached
                            carrier=self.carrier,  # Name of the carrier of the technology
                            p_nom=tot,  # Nominal power (MW)
                            p_min_pu=(poweroff / (tot * 1000000)).tolist(),  # Minimum output
                            p_max_pu=(poweroff / (tot * 1000000)).tolist(),  # Maximum output
                            marginal_cost=functions.calculate_marginal_costs(self.fuelcost,
                                                                             self.variableom,
                                                                             self.efficiency),
                            capital_cost=functions.calculate_capital_costs(self.discountrate, self.lifetime,
                                                                           self.fixedOM_p,
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
        self.CAPEX = data.loc[data["technology"] == self.filiere].squeeze()["nominal investment variable"]
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
                network.data_dir + '/registre-des-installations-de-production-et-de-stockage.csv',
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
                        )

    def constraint_disp(self, n, model, snap, xa, ext):
        """Constraint to define the disponibility of power generation facilities"""

        def disp(m, k):
            if ext:
                return sum(m.variables["Generator-p"][j, k] for j in list(snap)) - snap.size * self.cf * m.variables["Generator-p_nom"][k] <= 0
            else:
                return sum(m.variables["Generator-p"][j, k] for j in list(snap)) <= snap.size * self.cf * n.generators.p_nom[k]

        model.add_constraints(disp, coords=(xa,), name="disp_" + self.filiere)

    def constraint_min_max(self, n, model, snap, xa, ext, spec):
        """
        Constraint to define the minimum and maximum yearly generation of power generation facilities
        :param n: network
        :param model: model
        :param snap: snapshots
        :param xa: xarray of the technologies
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        :param spec: str, if only a min or a max constrain must be added
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
            if spec == "min":
                model.add_constraints(
                    sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) >= self.min_y,
                    name="limit1_" + str(self.filiere))
            elif spec == "max":
                model.add_constraints(
                    sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) <= self.max_y,
                    name="limit2_" + str(self.filiere))
            else:
                model.add_constraints(
                    sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) >= self.min_y,
                    name="limit1_" + str(self.filiere))
                model.add_constraints(
                    sum(model.variables["Generator-p"][j, i] for i in xa for j in list(snap)) <= self.max_y,
                    name="limit2_" + str(self.filiere))
