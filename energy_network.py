import os
import time
import pypsa
import ast
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from math import *
from pypsa.plot import add_legend_patches, add_legend_circles, add_legend_lines
import functions_used as functions
import additional_constraints as cs
from electrical_grid import ElectricalGrid, ExistingStorages, AdditionalStorages
from electricity_production import PV, Wind, BaseProduction, ETM
from electrical_demand import ElectricalDemand
from hydrogen_elements import H2Chain, H2Demand


class EnergyNetwork(pypsa.Network):
    """
    Represents an energy network.

    This class extends the `pypsa.Network` class and provides additional functionality specific to energy network simulations.
    """

    def __init__(self, snapshots):
        """
        Initializes an instance of the EnergyNetwork class.

        This constructor initializes the instance by calling the constructor of the base class `pypsa.Network`.
        """

        super().__init__(override_component_attrs=self.get_component_attrs())

        self.cons = None
        self.set_snapshots(snapshots)
        self.year = self.snapshots.year.unique().values[0]
        self.scenario = None
        self.data = None
        self.data_dir = None
        self.eleclines = None
        self.stations = None
        self.horizon = None
        self.climate_scenario = None
        self.vehicles_scenario = None
        self.buses_scenario = None
        self.h2_places = None


    def get_component_attrs(self):
        """
        Get the component attributes with custom additions

        :return: pypsa.descriptors.Dict, component attributes
        """
        # Add custom attributes to specific components
        attributes_to_add = {
            "env_f": ["float", "kgCO2eq/MW", 0.0, "fixed environmental impact", "Input (optional)"],
            "env_v": ["float", "kgCO2eq/MWh", 0.0, "variable environmental impact", "Input (optional)"],
            "water_f": ["float", "L/MW", 0.0, "fixed water consumption", "Input (optional)"],
            "water_v": ["float", "L/MWh", 0.0, "variable water consumption", "Input (optional)"],
            "base_CAPEX": ["float", "€", 0.0, "base cost of CAPEX for some generators", "Input (optional)"]
        }
        components_to_update = ['Generator', 'Link', 'Store', 'StorageUnit', 'Line']

        component_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})

        # Update the component attributes with custom additions
        for component in components_to_update:
            attrs = component_attrs[component]
            for attr_name, attr_value in attributes_to_add.items():
                attrs.loc[attr_name] = attr_value

        return component_attrs


    def import_network(self, data_dir, h2, h2bus, h2station, h2disp, ext, selfsufficiency, aircraft, marine, multiyear, yeardata):
        """
        Import and definition of the energy network

        :param data_dir: str, path for the data files
        :param h2: str, hydrogen scenario simulated
        :param h2bus: str, hydrogen bus scenario simulated
        :param h2station: int, number of stations for the hydrogen bus scenario simulated
        :param h2disp: int, number of dispensers for the hydrogen bus scenario simulated
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        :param selfsufficiency: bool, switch to optimize aircraft fuels
        :param aircraft: bool, switch to consider alternative fuels for the aviation sector
        :param marine: bool, switch to consider alternative fuels for the maritime sector
        :param multiyear: bool, switch to run multiple years
        :param yeardata: int, year to consider data for simulations with multiple years
        :return: tuple of already used and new electricity production technologies
        """
        # Store processed data in instance variables
        if multiyear:
            self.year = yeardata
        self.data, lines = functions.import_from_excel_folder(data_dir, self.year)
        self.data_dir = data_dir
        self.eleclines = lines

        # Extract scenario and consumption data from network information
        self.scenario = str(self.data['network'].at['production scenario', self.data['network'].columns.dropna()[0]])
        self.cons = str(self.data['network'].at['consumption scenario', self.data['network'].columns.dropna()[0]])
        self.climate_scenario = str(self.data['network'].loc['climate scenario'].dropna().values[0])
        self.vehicles_scenario = self.data['network'].loc['vehicles scenario'].dropna().values.flatten().tolist()

        # Extract buses scenario from network information
        buses = self.data['network'][self.data['network'].index == 'buses scenario'].dropna(axis=1)
        self.buses_scenario = [
            buses['List 1'].values[0],
            ast.literal_eval(buses['List 2'].values[0]),
            ast.literal_eval(buses['List 3'].values[0])
        ]

        self.horizon = self.snapshots.tolist()

        self.import_meteo_data(multiyear)
        self.import_carriers()

        # Import buses and lines into the electrical grid
        electrical_grid = ElectricalGrid(self)
        electrical_grid.import_buses()
        electrical_grid.import_lines()
        if aircraft:
            electrical_grid.import_bus_aircraft()
            electrical_grid.import_lines_aircraft()

        # Import existing storages
        existing_storages = ExistingStorages(self)
        existing_storages.import_storages()

        # Import generators
        self.import_generators(ext)

        # Update generator max production values for a year
        # Maximum production is given for a specific capacity installed (depending on data availability).
        # The following lines update maximum production according to capacity installed (linear relation).
        if not ext:
            self.update_hydraulic_generator_values()
            for _, row in self.data['generator_data'][['max_capa', 'technology', 'max_year', 'min_year']].dropna().iterrows():
                if row['technology'] == 'Biomasse':
                    self.data['generator_data'].loc[
                        self.data['generator_data']['technology'] == row['technology'], 'max_year'] = \
                        row['max_year'] * self.generators['p_nom'].filter(regex='Biomasse|Bagasse|Bioénergie|bioéthanol').sum() / row[
                            'max_capa']
                    self.data['generator_data'].loc[
                        self.data['generator_data']['technology'] == row['technology'], 'min_year'] = \
                        row['min_year'] * self.generators['p_nom'].filter(regex='Biomasse|Bagasse|Bioénergie|bioéthanol').sum() / row[
                            'max_capa']
                else:
                    self.data['generator_data'].loc[
                        self.data['generator_data']['technology'] == row['technology'], 'max_year'] = \
                        row['max_year'] * self.generators['p_nom'].filter(regex=row['technology']).sum() / row['max_capa']
                    self.data['generator_data'].loc[
                        self.data['generator_data']['technology'] == row['technology'], 'min_year'] = \
                        row['min_year'] * self.generators['p_nom'].filter(regex=row['technology']).sum() / row['max_capa']

        # Import electrical demand and additional batteries
        electrical_demand = ElectricalDemand(self)
        electrical_demand.import_demand(multiyear)
        additional_storages = AdditionalStorages(self)
        additional_storages.import_storages()

        if aircraft:
            ratio = None
            if selfsufficiency:
                if os.path.exists('/home/afrancoi/Downloads/Ratio aviation.xlsx'):
                    ratio = pd.read_excel('/home/afrancoi/Downloads/Ratio aviation.xlsx', header=0)
                    ratio['Month'] = pd.to_datetime(str(self.year) + '-' + ratio['Month'].astype(str), format='%Y-%m')
                    ratio = ratio.set_index('Month')
                    ratio = ratio.reindex(self.horizon, method='ffill')
                else:
                    raise ValueError('ERROR: no ratio for aircraft file found.')
            electrical_demand.import_aircraft_elec_demand(rt=ratio)

            # Aircraft fuels produced with additional power plant :
            self.add("Generator",
                     "generator joker Roland Garros airport",
                     bus="electricity bus Roland Garros airport",
                     carrier="joker",
                     p_nom_extendable=True,
                     # p_nom_max=10000,
                     marginal_cost=1000,
                     capital_cost=1e10,
                     )

            # Aircraft fuels produced with additional power plants at each substation :
            # for i in self.buses.index:
            #     if "electricity bus" in i:
            #         self.add("Generator",
            #                  "generator joker " + i[16:],
            #                  bus=i,
            #                  carrier="joker",
            #                  p_nom_extendable=True,
            #                  p_nom_max=100,
            #                  marginal_cost=1000,
            #                  capital_cost=1e10,
            #                  )

            # Aircraft fuels produced with offshore wind, installed capacity optimised :
            # Wind(self.data["generator_data"], "offshore").import_wind_offshore_ext(self, self.data["wind"]['Digue'],
            #                                                                        self.data["meteo_t"]['Digue'],
            #                                                                        'Roland Garros airport')

        if marine:
            electrical_demand.import_maritime_elec_demand()

        # Import hydrogen technologies according to the scenario passed
        postes = self.data["postes"].index
        # Hydrogen technologies will be installed to every substation by default. Some substations can be removed to
        # the list, so that no new technology is installed:
        postes_removed = ['Takamaka']
        # postes_removed.extend(['Dattiers', 'Moufia', 'Digue', 'St Pierre', 'St Paul', 'Langevin', 'Le Bras de la Plaine'])
        postes = postes.drop(postes_removed)

        h2_chain_main = H2Chain(self.data, postes)
        if h2 == "stock":  # Electrolyser, storage lp and fuel cell at every substation
            h2_chain_main.import_electrolyser(self)
            h2_chain_main.import_h2_storage_lp(self)
            h2_chain_main.import_fc(self)
        elif h2 != 'None':  # Refueling stations on the substations concerned
            h2_places_buses = pd.Series([], dtype=str)
            h2_places_train = pd.Series([], dtype=str)
            if "buses" in h2:
                h2_places_buses = self.data['load_car'][self.scenario][:h2station]
                # h2_places_buses = self.data['load_car'][100][:h2station]
                h2_demand = H2Demand(h2_places_buses, self.data_dir)
                h2_demand.import_h2_buses(self, h2bus, h2disp)
            if "train" in h2:
                h2_places_train = self.data['load_train'].index.to_series()
                h2_demand = H2Demand(h2_places_train, self.data_dir)
                h2_demand.import_h2_train(self)
            self.h2_places = pd.concat([h2_places_buses, h2_places_train]).drop_duplicates().reset_index(drop=True)  # stations are pooled
            h2_chain = H2Chain(self.data, self.h2_places)
            if "stock" in h2:  # Electrolyser, storage lp and fuel cell at every substation
                h2_chain_main.import_electrolyser(self)
                h2_chain_main.import_h2_storage_lp(self)
                h2_chain_main.import_fc(self)
            else:  # Electrolyser only at refueling station
                h2_chain.import_electrolyser(self)
            h2_chain.import_compressor(self)
            h2_chain.import_h2_storage_hp(self)
            h2_chain.import_expander(self)

        if aircraft:
            h2_demand = H2Demand("", self.data_dir)
            h2_demand.import_aviation_hydrogen_demand(self, rt=ratio)

        if marine:
            h2_demand = H2Demand("", self.data_dir)
            h2_demand.import_maritime_hydrogen_demand(self)

        return (
            ast.literal_eval(self.data['network'].at['generation base', 'List 1']),
            ast.literal_eval(self.data['network'].at['generation new', 'List 1'])
        )


    def import_meteo_data(self, multiyear):
        """
        Import meteorological data

        :param multiyear: bool, switch to run multiple years
        """

        if multiyear:
            self.data['meteo_r'] = pd.read_csv(f"{self.data_dir}/rayonnement_stations_corrige.csv",
                                               sep=',', encoding='latin-1', index_col=0, parse_dates=True)
            self.data['meteo_r'] = self.data['meteo_r'][(self.data['meteo_r'].index.year >= self.snapshots.year.unique().values[0]) & (self.data['meteo_r'].index.year <= self.snapshots.year.unique().values[-1])]
            self.data['meteo_r'] = self.data['meteo_r'][(self.data['meteo_r'].index.month != 2) | (self.data['meteo_r'].index.day != 29)]

            self.data['meteo_t'] = pd.read_csv(f"{self.data_dir}/temperature_stations_corrige.csv",
                                               sep=',', encoding='latin-1', index_col=0, parse_dates=True)
            self.data['meteo_t'] = self.data['meteo_t'][(self.data['meteo_t'].index.year >= self.snapshots.year.unique().values[0]) & (self.data['meteo_t'].index.year <= self.snapshots.year.unique().values[-1])]
            self.data['meteo_t'] = self.data['meteo_t'][(self.data['meteo_t'].index.month != 2) | (self.data['meteo_t'].index.day != 29)]

            self.data['wind'] = pd.read_csv(f"{self.data_dir}/vent_stations_corrige.csv",
                                            sep=',', encoding='latin-1', index_col=0, parse_dates=True)
            self.data['wind'] = self.data['wind'][(self.data['wind'].index.year >= self.snapshots.year.unique().values[0]) & (self.data['wind'].index.year <= self.snapshots.year.unique().values[-1])]
            self.data['wind'] = self.data['wind'][(self.data['wind'].index.month != 2) | (self.data['wind'].index.day != 29)]

            self.data['rain'] = pd.read_csv(
                f"{self.data_dir}/BRIO/Precipitations/Prec_scena{self.climate_scenario}_moy{str(self.year)[-2:]}.csv"
            )


        else:
            self.data['meteo_r'] = pd.read_csv(f"{self.data_dir}/rayonnement_tmy_{str(self.year)}.csv",
                                               sep=',', encoding='latin-1', index_col=0)

            self.data['meteo_t'] = pd.read_csv(f"{self.data_dir}/BRIO/T_{str(self.year)[-2:]}_{self.climate_scenario}.csv",
                                               sep=',', encoding='latin-1', index_col=0)

            self.data['wind'] = pd.read_csv(f"{self.data_dir}/vent_construit_AROME.csv",
                                            encoding='latin-1', index_col=0)
            self.data['wind'].sort_index(inplace=True)

            self.data['rain'] = pd.read_csv(
                f"{self.data_dir}/BRIO/Precipitations/Prec_scena{self.climate_scenario}_moy{str(self.year)[-2:]}.csv"
            )


        self.data['meteo_r'].index = self.horizon
        self.data['meteo_t'].index = self.horizon
        self.data['wind'].index = self.horizon

        # Can be used for wind power models
        self.data['meteo_t_corrige'] = self.data['meteo_t'].copy()
        self.data['wind corrige'] = self.data['wind'].copy()
        self.data['rho corrige'] = self.data['wind'].copy()

        if not {'timec', 'lon', 'lat', 'pr_corr'}.issubset(self.data['rain'].columns.values.tolist()):
            raise ValueError('ERROR: rainfall file not formatted.')
        self.data['rain']['timec'] = pd.to_datetime(self.data['rain']['timec'], format='%Y-%m-%d %H:%M:%S')


    def update_hydraulic_generator_values(self):
        """
        Update hydraulic generator values

        :param data: Data containing generator information
        """
        hydrau = self.generators[self.generators.index.str.contains("Hydraulique")].index.to_list()
        power_file = self.generators.p_nom[hydrau].rename(lambda x: x[:-12]).to_frame()

        precs = functions.create_weighted_rainfall(self.data['rain'], power_file, self.data['postes'])

        val_min = (precs * 3.67e-5 + 0.183) * power_file.sum() * 8760  # Values studied for La Réunion
        self.data['generator_data'].loc[self.data['generator_data']['technology'] == 'Hydraulique', 'min_year'] = val_min[0]

        val_max = (precs * 3.67e-5 + 0.373) * power_file.sum() * 8760  # Values studied for La Réunion
        self.data['generator_data'].loc[self.data['generator_data']['technology'] == 'Hydraulique', 'max_year'] = val_max[0]


    def import_carriers(self):
        """
        Function to import the different energy carriers involved
        """
        self.madd("Carrier", self.data["carrier"]["name"].tolist(), color=self.data["carrier"]["color"].tolist())


    def import_generators(self, ext):
        """
        Function to import the generators of the energy system
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        """
        generators = self.data["generator"].sort_values(by=["Poste source", "Filière"])  # Values are sorted by station and carrier
        generators = generators.reset_index()
        total_capa = 0

        for index, row in generators.iterrows():  # Loop in the file with all technologies for electricity production
            ps = row["Poste source"]
            fil = row["Filière"]
            total_capa += row["Puissance installée (kW)"]
            # PyPSA capacity unit is MW. EDF capacity unit is kW. Capacities of the same technology of the same station are added
            if index == generators.shape[0] - 1 or generators.loc[index + 1, "Poste source"] != ps or generators.loc[
                index + 1, "Filière"] != fil:
                # We add elements to our model when the last element of the file is reached or the last element of the station/of the technology

                if "PV+stockage" in fil:
                    total_capa = 0

                elif "PV" in fil:
                    PV(self.data["generator_data"]).import_pv(self, round(total_capa, 2) / 1000, self.data["meteo_t"][ps], self.data["meteo_r"][ps], ps, ext)

                elif fil == "Eolien":
                    Wind(self.data["generator_data"], "onshore").import_wind(self, total_capa/1000, self.data["wind"][ps], self.data["meteo_t"][ps], ps, ext)

                elif fil == "Eolien offshore":
                    Wind(self.data["generator_data"], "offshore").import_wind(self, total_capa/1000, self.data["wind"]["Offshore"], self.data["meteo_t"][ps], ps, ext)

                elif fil == "ETM":
                    ETM(self.data["generator_data"]).import_etm(self, total_capa/1000, ps)

                else:
                    BaseProduction(self.data["generator_data"], fil).import_base(self, round(total_capa, 2) / 1000, ps, ext)
                total_capa = 0


    def optimization(self, solver, solver_options, h2, sec_new, obj, water, ext, multiyear):
        """
        Function for the creation of the optimisation problem and its solving
        :param solver: str, solver used
        :param solver_options: dict, keyword arguments used by the solver
        :param h2: str, hydrogen scenario simulated
        :param sec_new: list, production sectors newly installed (may only work for Reunion)
        :param obj: str, type of the optimisation
        :param water: float, limit for water consumption
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        :param multiyear: bool,
        :return: costs and environmental impact
        """
        print("INFO: creating '{}' optimisation...".format(obj))
        tic = time.time()
        model = self.optimize.create_model(transmission_losses=1)

        # Bounds directly on the variables for nominal power
        # (it can reduce calculation time, but it has not been tested precisely)
        if ext:
            model.variables["Generator-p_nom"].lower = xr.DataArray(
                self.generators['p_nom_min'][self.get_extendable_i('Generator')].tolist(),
                coords=(self.get_extendable_i('Generator'),))
            model.variables["Generator-p_nom"].upper = xr.DataArray(
                self.generators['p_nom_max'][self.get_extendable_i('Generator')].tolist(),
                coords=(self.get_extendable_i('Generator'),))
        model.variables["Line-s_nom"].lower = xr.DataArray(
            self.lines['s_nom_min'][self.get_extendable_i('Line')].tolist(),
            coords=(self.get_extendable_i('Line'),))
        model.variables["Link-p_nom"].lower = xr.DataArray(
            self.links['p_nom_min'][self.get_extendable_i('Link')].tolist(),
            coords=(self.get_extendable_i('Link'),))
        model.variables["Store-e_nom"].lower = xr.DataArray(
            self.stores['e_nom_min'][self.get_extendable_i('Store')].tolist(),
            coords=(self.get_extendable_i('Store'),))


        # H2Chain(self.data, pd.Series(['Marquet'])).constraint_prodsup_bus(self, model, self.horizon)
        # H2Chain(self.data, pd.Series(['Roland Garros airport'])).constraint_prodsup_bus(self, model, self.horizon)
        # H2Chain(self.data, pd.Series(['Roland Garros airport'])).constraint_cyclic_soc(self, model, self.horizon, 10)
        # H2Chain(self.data, pd.Series(['Roland Garros airport'])).constraint_minimal_soc(self, model, self.horizon, 1, True, True)
        # H2Chain(self.data, pd.Series(self.data['postes'].index.tolist())).constraint_prodsup_bus(self, model, self.horizon)
        # model.add_constraints(sum(model.variables["Generator-p"][j, 'generator joker Roland Garros airport'] for j in list(self.horizon)) - 8760 * 0.8 * model.variables["Generator-p_nom"]['generator joker Roland Garros airport'] <= 0, name="charge_joker")


        # Constraints for the definition of the hydrogen chain with hydrogen demand
        if (h2 != "stock") and (h2 != "None"):
            H2Chain(self.data, self.h2_places).constraint_prodsup_bus(self, model, self.horizon)
            H2Chain(self.data, self.h2_places).constraint_cyclic_soc(self, model, self.horizon, 10)
            H2Chain(self.data, self.h2_places).constraint_minimal_soc(self, model, self.horizon, 1, False, False)

        # if h2 == "stock":
        #     postes = self.data["postes"].index
        #     postes = postes.drop('Takamaka')  # contrainte parc national
            # postes = postes.drop('Dattiers')
            # postes = postes.drop('Moufia')
            # postes = postes.drop('Digue')
            # postes = postes.drop('St Pierre')
            # postes = postes.drop('St Paul')
            # postes = postes.drop('Langevin')
            # postes = postes.drop('Le Bras de la Plaine')
            # H2Chain(self.data, pd.Series(postes.to_list())).constraint_prodsup_bus(self, model, self.horizon)
            # H2Chain(self.data, pd.Series(self.data['postes'].index.tolist())).constraint_prodlong_bis(self, model, self.horizon)
            # H2Chain(self.data, pd.Series(self.data['postes'].index.tolist())).constraint_prodsup_bus(self, model, self.horizon)

        # model.add_constraints(model.variables['Generator-p_nom']['Aviation electricity export'] - model.variables['Generator-p_nom']['Aviation hydrogen export'] <= 0, name='equality_aviation_1')
        # model.add_constraints(
        #     model.variables['Generator-p_nom']['Aviation electricity export'] - model.variables['Generator-p_nom'][
        #         'Aviation hydrogen export'] >= 0, name='equality_aviation_2')


        # Constraints for the definition of the additional storages
        AdditionalStorages(self).constraints_additionnal_battery(self, model)

        # Constraint for limiting total storage capacity installed
        # v_store, c_store = cs.limit_storage(self, model)
        # model.add_constraints(v_store <= 2200 - c_store, name="limit_store")

        # Constraints for the definition of the disponibility and annual limit of electricity generation technologies

        # HYDROELECTRICITY
        hydrau = self.generators[self.generators.index.str.contains("Hydraulique")].index.to_list()
        # If disponibility constraint (code not constructed for multiyear):
        hydrau_xa = pd.Series(hydrau).to_xarray().rename({'index': 'hydrau'})
        # BaseProduction(self.data["generator_data"], "Hydraulique").constraint_disp(self, model, self.snapshots, hydrau_xa, ext)
        # If maximum production per year:
        if multiyear:
            for i in self.snapshots.year.unique():
                BaseProduction(self.data["generator_data"], "Hydraulique").constraint_min_max(self, model,
                                                                                              self.snapshots[self.snapshots.year == i],
                                                                                              hydrau, ext, spec=None)
        else:
            BaseProduction(self.data["generator_data"], "Hydraulique").constraint_min_max(self, model, self.snapshots,
                                                                                          hydrau, ext, spec=None)

        # BIOENERGY
        bioenergie = self.generators[self.generators.index.str.contains("Bioénergie")].index.to_list()
        bioenergie_xa = pd.Series(bioenergie).to_xarray().rename({'index': 'bioenergie'})
        if multiyear:
            for i in self.snapshots.year.unique():
                BaseProduction(self.data["generator_data"], "Bioénergie").constraint_disp(self, model,
                                                                                          self.snapshots[self.snapshots.year == i],
                                                                                          bioenergie_xa, ext)
        else:
            BaseProduction(self.data["generator_data"], "Bioénergie").constraint_disp(self, model, self.snapshots,
                                                                                      bioenergie_xa, ext)


        bioethanol = self.generators[self.generators.index.str.contains("TAC bioéthanol")].index.to_list()
        # If disponibility constraint (code not constructed for multiyear):
        bioethanol_xa = pd.Series(bioethanol).to_xarray().rename({'index': 'bioethanol'})
        # BaseProduction(self.data["generator_data"], "TAC bioéthanol").constraint_disp(self, model, self.snapshots, bioethanol_xa, ext)
        # If maximum production per year:
        if multiyear:
            for i in self.snapshots.year.unique():
                BaseProduction(self.data["generator_data"], "TAC bioéthanol").constraint_min_max(self, model,
                                                                                                 self.snapshots[self.snapshots.year == i],
                                                                                                 bioethanol, ext, spec=None)
        else:
            BaseProduction(self.data["generator_data"], "TAC bioéthanol").constraint_min_max(self, model, self.snapshots,
                                                                                     bioethanol, ext, spec=None)

        # BIOMASS
        biomasse = self.generators[self.generators.index.str.contains("Biomasse")].index.to_list()
        bagasse = self.generators[self.generators.index.str.contains("Bagasse")].index.to_list()
        bioimport = self.generators[self.generators.index.str.contains("import")].index.to_list()
        if multiyear:
            for i in self.snapshots.year.unique():
                BaseProduction(self.data["generator_data"], "Bagasse").constraint_min_max(self, model,
                                                                                          self.snapshots[self.snapshots.year == i],
                                                                                          bagasse, ext, spec=None)
                BaseProduction(self.data["generator_data"], "Biomasse").constraint_min_max(self, model,
                                                                                           self.snapshots[self.snapshots.year == i],
                                                                                           biomasse + bagasse, ext, spec="max")
        else:
            BaseProduction(self.data["generator_data"], "Bagasse").constraint_min_max(self, model, self.snapshots,
                                                                                      bagasse, ext, spec=None)
            BaseProduction(self.data["generator_data"], "Biomasse").constraint_min_max(self, model, self.snapshots,
                                                                                       biomasse + bagasse, ext,
                                                                                       spec="max")
            # BaseProduction(self.data["generator_data"], "Biomasse import").constraint_min_max(self, model, self.snapshots, bioimport, ext, spec="min")


        for i in sec_new:
            data_list = self.generators.index.str.contains(i)
            if data_list.any() and i != "ETM":
                index_list = self.generators[data_list].index.to_list()
                # If disponibility constraint (code not constructed for multiyear):
                index_list_xa = pd.Series(index_list).to_xarray().rename({'index': i})
                # BaseProduction(self.data["generator_data"], i).constraint_disp(self, model, self.snapshots, index_list_xa, False)
                # If maximum production per year:
                if multiyear:
                    for j in self.snapshots.year.unique():
                        BaseProduction(self.data["generator_data"], i).constraint_min_max(self, model,
                                                                                          self.snapshots[self.snapshots.year == j],
                                                                                          index_list, False, spec=None)
                else:
                    BaseProduction(self.data["generator_data"], i).constraint_min_max(self, model, self.snapshots,
                                                                                      index_list, False, spec=None)


        if ext:
            # Only one potential for geothermal energy and OTEC: all or nothing (MILP /!\)
            geothermal = self.generators[self.generators.index.str.contains("Geothermie")].index.to_list()
            model.add_variables(name="x_geothermal", binary=True)
            model.add_constraints(model.variables["Generator-p_nom"][geothermal[0]].to_linexpr() -
                                  model.variables['x_geothermal'] * self.generators['p_nom_max'][geothermal[0]] <= 0,
                                  name="p_geothermal_1")
            model.add_constraints(model.variables["Generator-p_nom"][geothermal[0]].to_linexpr() -
                                  model.variables['x_geothermal'] * self.generators['p_nom_max'][geothermal[0]] >= 0,
                                  name="p_geothermal_2")
            etm = self.generators[self.generators.index.str.contains("ETM")].index.to_list()
            model.add_variables(name="x_etm0", binary=True)
            model.add_variables(name="x_etm1", binary=True)
            model.add_constraints(model.variables["Generator-p_nom"][etm[0]].to_linexpr() - model.variables['x_etm0'] * self.generators['p_nom_max'][etm[0]] <= 0, name="p_etm_01")
            model.add_constraints(model.variables["Generator-p_nom"][etm[0]].to_linexpr() - model.variables['x_etm0'] * self.generators['p_nom_max'][etm[0]] >= 0, name="p_etm_02")
            model.add_constraints(model.variables["Generator-p_nom"][etm[1]].to_linexpr() - model.variables['x_etm1'] * self.generators['p_nom_max'][etm[1]] <= 0, name="p_etm_11")
            model.add_constraints(model.variables["Generator-p_nom"][etm[1]].to_linexpr() - model.variables['x_etm1'] * self.generators['p_nom_max'][etm[1]] >= 0, name="p_etm_12")


        # Constraint for water consumption
        # v_water, c_water = cs.impact_constraint(self, model, 'water')
        # model.add_constraints(v_water <= water - c_water, name="water_impact")

        # Addition to the objective function of the OTEC CAPEX base (among others)
        objective_constant_2 = sum(self.generators["base_CAPEX"][i] for i in self.generators.index.tolist())
        object_const_2 = model.add_variables(objective_constant_2, objective_constant_2, name="objective_constant_2")
        model.objective += (-1 * object_const_2)

        toc = time.time()
        print("INFO: creating the model took {} minutes.".format((toc - tic) / 60))
        tic = time.time()

        if obj == 'multi':
            if not os.path.exists("/home/afrancoi/PyPSA/Résultats/MOO test/cost min"):
                raise ValueError('ERROR: no directory ready to store results.')
            cost_list = []
            env_list = []

            print('START OF THE MULTI-OBJECTIVE OPTIMISATION : ECONOMIC MINIMUM PERFORMED...')
            self.optimize.solve_model(solver_name=solver, **solver_options)
            if not self.model.status == 'ok':
                raise ValueError('ERROR: optimization is infeasible, results cannot be plotted.')
            cost_list.append(cs.impact_result(self, 'cost'))
            env_list.append(cs.impact_result(self, 'env'))
            print('ECONOMIC MINIMUM:', cost_list[0])
            print('ENVIRONMENTAL MAXIMUM:', env_list[0])
            print('Water consumption:', cs.impact_result(self, 'water'))
            print('Total of storages (MWh):', self.stores.groupby(['carrier']).e_nom_opt.sum())
            enr_inter, operation = self.generator_data()
            self.export_to_csv_folder('/home/afrancoi/PyPSA/Résultats/MOO test/cost min')
            print('Network successfully exported.')

            # Update of objective function to have the environmental optimum
            obj_stock = model.objective
            model.objective = cs.impact_constraint(self, model, 'env')[0].to_linexpr()
            print('ENVIRONMENTAL MINIMUM PERFORMED...')
            self.optimize.solve_model(solver_name=solver, **solver_options)
            if not self.model.status == 'ok':
                raise ValueError('ERROR: optimization is infeasible, results cannot be plotted.')

            max_cost = cs.impact_result(self, 'cost')
            min_env = cs.impact_result(self, 'env')
            print('ECONOMIC MAXIMUM:', max_cost)
            print('ENVIRONMENTAL MINIMUM:', min_env)
            print('Water consumption:', cs.impact_result(self, 'water'))
            print('Total of storages (MWh):', self.stores.groupby(['carrier']).e_nom_opt.sum())
            enr_inter, operation = self.generator_data()
            self.export_to_csv_folder('/home/afrancoi/PyPSA/Résultats/MOO test/env min')
            print('Network successfully exported.')

            # Start of the multi-objective optimisation with epsilon-constraint method
            step = (env_list[0] - min_env) / 4  # Number of iterations (+1) arbitrarily set
            model.objective = obj_stock  # Going back to economic optimum
            for i in list(range(floor(env_list[0]), floor(min_env), -floor(step)))[1:-1]:
                # Number of the iteration
                a = list(range(floor(env_list[0]), floor(min_env), -floor(step)))[1:-1].index(i) + 1
                # Constraints for environmental impact within multi-objective optimisation
                v_env, c_env = cs.impact_constraint(self, model, 'env')
                model.add_constraints(v_env <= i - c_env, name="env_impact")
                print('PERFORMING OPTIMISATION {} / 3'.format(a))
                self.optimize.solve_model(solver_name=solver, **solver_options)
                if not self.model.status == 'ok':
                    raise ValueError('ERROR: optimization is infeasible, results cannot be plotted.')

                cost_list.append(cs.impact_result(self, 'cost'))
                env_list.append(cs.impact_result(self, 'env'))
                model.constraints.remove(name="env_impact")
                print('ECONOMIC OPTIMUM:', cs.impact_result(self, 'cost'))
                print('ENVIRONMENTAL OPTIMUM:', cs.impact_result(self, 'env'))
                print('Water consumption:', cs.impact_result(self, 'water'))
                print('Total of storages (MWh):', self.stores.groupby(['carrier']).e_nom_opt.sum())
                enr_inter, operation = self.generator_data()
                self.export_to_csv_folder('/home/afrancoi/PyPSA/Résultats/MOO test/Pareto front ' + str(a))
                print('Network successfully exported.')

            cost_list.append(max_cost)
            env_list.append(min_env)

            return cost_list, env_list


        else:
            if obj == 'env':
                model.objective = cs.impact_constraint(self, model, obj)[0].to_linexpr()

            self.optimize.solve_model(solver_name=solver, **solver_options)

            if not self.model.status == 'ok':
                raise ValueError('ERROR: optimization was not successful, results cannot be plotted.')

            toc = time.time()
            print("INFO: solving took {} minutes.".format((toc - tic)/60))

            return cs.impact_result(self, 'cost'), cs.impact_result(self, 'env'), cs.impact_result(self, 'water')


    def plot_network(self, status, stor, ely, fc):
        """
        Plot the network before and after optimization.

        :param status: The status of the network to be plotted. Valid values are 'initial' or 'final'.
        :type status: str

        :param stor: Indicates whether to plot the locations and sizes of the storages after optimization.
        :type stor: bool

        :param ely: Indicates whether to plot the locations and sizes of the electrolyzers after optimization.
        :type ely: bool

        :param fc: Indicates whether to plot the locations and sizes of the fuel cells after optimization.
        :type fc: bool

        :return: None
        :rtype: None
        """

        legend_kwargs = {"loc": "upper left", "frameon": False}
        legend_circles_dict = {"bbox_to_anchor": (1, 0.8), "labelspacing": 2.5, **legend_kwargs}
        line_sizes = [26, 44.7]  # in MVA

        if status == 'initial':
            bus_sizes = [50, 100]  # in MW
            unit = 'MW'
            gen = self.generators.groupby(['bus', 'carrier']).p_nom.sum()
            lines = self.lines.s_nom / 10
            title = "Reunion's electricity grid before optimization"
            legend1 = self.carriers.loc[self.generators.carrier.unique()]['color']
            legend2 = self.generators.carrier.unique()
            save = "network_map.png"

        elif status == 'final':
            if stor:
                bus_sizes = [50, 100]  # in MWh
                unit = 'MWh'
                self.stores['elec bus'] = 0
                for i in self.stores.index:
                    if "additional" in i:
                        self.stores['elec bus'][i] = 'electricity bus ' + i[19:]
                    elif "hydrogen" in i:
                        self.stores['elec bus'][i] = 'electricity bus ' + i[20:]
                gen = self.stores[self.stores['elec bus'] != 0].groupby(['elec bus', 'carrier']).e_nom_opt.sum()
                lines = self.lines.s_nom_opt / 10
                title = "Reunion's electricity grid after optimization - storages"
                legend1 = self.carriers.loc[self.stores.carrier.unique()]['color']
                legend2 = self.stores.carrier.unique()
                save = "network_map_stor.png"

            elif ely:
                bus_sizes = [50, 100]  # in MW
                unit = 'MW'
                gen = self.links.loc[self.links[self.links.index.str.contains("electrolyser")].index][['bus0', 'p_nom_opt']].set_index('bus0').squeeze()
                lines = self.lines.s_nom_opt / 10
                title = "Reunion's electricity grid before optimization - electrolysers"
                legend1 = self.carriers.loc[self.links.carrier.unique()]['color']
                legend2 = self.links.carrier.unique()
                save = "network_map_ely.png"

            elif fc:
                bus_sizes = [50, 100]  # in MW
                unit = 'MW'
                gen = self.links.loc[self.links[self.links.index.str.contains("fuel cell")].index][['bus1', 'p_nom_opt']].set_index('bus1').squeeze()
                lines = self.lines.s_nom_opt / 10
                title = "Reunion's electricity grid before optimization - fuel cells"
                legend1 = self.carriers.loc[self.links.carrier.unique()]['color']
                legend2 = self.links.carrier.unique()
                save = "network_map_fc.png"

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Plot network
        self.plot(ax=ax, title=title, color_geomap=True, bus_sizes=gen / 3e5, line_widths=lines,
                branch_components=['Line'], boundaries=[55.1115043336971, 55.97942417175307, -20.843415164623533, -21.424694661983377])

        # Add legends
        add_legend_patches(
            ax,
            legend1,
            legend2,
            legend_kw={"bbox_to_anchor": (1, 0), **legend_kwargs, "loc": "lower left"}
        )
        add_legend_circles(
            ax,
            [s / 3e5 for s in bus_sizes],
            [f"{s} " + unit for s in bus_sizes],
            legend_kw=legend_circles_dict
        )
        add_legend_lines(
            ax,
            [s / 10 for s in line_sizes],
            [f"{s} MVA" for s in line_sizes],
            patch_kw={'color': 'rosybrown'},
            legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs}
        )

        fig.tight_layout()
        fig.savefig(save, bbox_inches="tight", dpi=300)
        plt.show()


    def generator_data(self):
        """
        Function for the plot + information about the electricity mix after optimisation
        :return: dataframe with hourly intermittent rate
        """
        gen = self.generators_t.p.groupby(self.generators.carrier, axis=1).sum()
        pow = self.generators.p_nom_opt.groupby(self.generators.carrier).sum()
        colors = self.carriers['color']

        fig = plt.figure()
        gen.sum().plot.pie(title='Electricity mix over the simulated year', autopct='%1.1f%%', colors=[colors[col] for col in gen.columns])
        fig.tight_layout()
        fig.savefig("electricity_mix.png", bbox_inches="tight", dpi=300)

        # Duration curve
        df1 = pd.concat([(gen[a] / pow[a]).sort_values(ascending=False).reset_index() for a in self.generators.carrier.unique()], axis=1)
        df1 = df1.drop(['snapshot'], axis=1)
        ax = df1.plot()
        ax.grid(True, linestyle='-.', which='both')
        ax.set_title('Duration curve per sector', fontsize=17)
        ax.set_ylabel("Normalised power (MW/MWmax)", fontsize=17)
        ax.set_xlabel("Cumulative hours for the year", fontsize=17)
        ax.tick_params(axis='both', which='both', labelsize=14)
        plt.tight_layout()
        plt.savefig("duration_curve.png", bbox_inches="tight", dpi=300)

        if 'coal' in gen.columns:
            print("RESULTS: {} MWh of fossil produced.".format(round(gen['coal'].sum())))
        print("RESULTS: {} MWh of hydroelectricity produced.".format(round(gen['water'].sum())))
        print("RESULTS: {} MWh of PV produced.".format(round(gen['solar'].sum())))
        print("RESULTS: {} MWh of onshore wind produced.".format(round(gen['wind onshore'].sum())))
        if 'wind offshore' in gen.columns:
            print("RESULTS: {} MWh of offshore wind produced.".format(round(gen['wind offshore'].sum())))
        print("RESULTS: {} MWh of biomass (global) produced.".format(round(gen['biogaz'].sum() + gen['bagasse'].sum() + gen['biomass'].sum())))
        if 'geothermal energy' in gen.columns:
            print("RESULTS: {} MWh of geothermal energy produced.".format(round(gen['geothermal energy'].sum())))
        if 'ocean thermal energy' in gen.columns:
            print("RESULTS: {} MWh of ETM produced.".format(round(gen['ocean thermal energy'].sum())))

        # Operating points
        stor = self.stores_t.p.sum(axis=1)
        stor[stor < 0] = 0
        df2 = pd.concat([gen.sum(axis=1) + stor + self.storage_units_t.p_dispatch.sum(axis=1),
                         (gen['wind onshore'] + gen['solar']) * 100 / gen.sum(axis=1)], axis=1)
        # ax = df2.plot(kind='scatter', x=0, y=1)
        # ax.grid(True, linestyle='-.', which='both')
        # ax.set_title('Diagram of the operating points of the electrical system', fontsize=17)
        # ax.set_ylabel("Intermittent renewable energy rate (%)", fontsize=17)
        # ax.set_xlabel("Production (including stored energy) (MW)", fontsize=17)
        # ax.tick_params(axis='both', which='both', labelsize=14)
        # plt.tight_layout()
        # plt.savefig("operating_points.png", bbox_inches="tight", dpi=300)

        # Duration curve of intermittent energies
        df3 = df2[1].sort_values(ascending=False).reset_index()
        df3 = df3.drop(['snapshot'], axis=1)
        ax = df3.plot(legend=False)
        ax.grid(True, linestyle='-.', which='both')
        ax.set_title('Duration curve of intermittent energy', fontsize=17)
        ax.set_ylabel("Intermittent energy rate (%)", fontsize=17)
        ax.set_xlabel("Cumulative hours for the year", fontsize=17)
        ax.tick_params(axis='both', which='both', labelsize=14)
        plt.tight_layout()
        plt.savefig("duration_curve_intermittent.png", bbox_inches="tight", dpi=300)

        # Operation over the year
        p_by_carrier = gen
        p_by_carrier['load'] = -self.loads_t.p_set.sum(axis=1)

        storage_disp = self.stores_t.p.loc[:, self.stores.carrier == 'electricity'].sum(axis=1)
        storage_disp[storage_disp < 0] = 0
        storage_disp = storage_disp + self.storage_units_t.p_dispatch.sum(axis=1)

        storage_stor = self.stores_t.p.loc[:, self.stores.carrier == 'electricity'].sum(axis=1)
        storage_stor[storage_stor > 0] = 0
        storage_stor = storage_stor - self.storage_units_t.p_store.sum(axis=1)

        stores = pd.concat([self.storage_units_t.state_of_charge, self.stores_t.e], axis=1).groupby(self.stores.carrier, axis=1).sum()

        p_by_carrier['battery charging'] = storage_stor
        p_by_carrier['battery discharging'] = storage_disp

        cols = np.append(colors.index.values[np.isin(colors.index.values, self.generators.carrier.unique())], ['load', 'battery discharging', 'battery charging'])
        p_by_carrier = p_by_carrier[cols]  # to sort the area graph
        c = [colors[col] for col in p_by_carrier.columns]
        fig, ax = plt.subplots(figsize=(12, 6))
        (p_by_carrier / 1e3).plot(kind="area", ax=ax, linewidth=0, color=c, alpha=0.7)
        (stores / 1e3).plot(ax=ax, linestyle='dashed')
        ax.legend(ncol=4, loc="upper left")
        ax.set_ylabel("GW")
        ax.set_xlabel("")
        ax.set_title('Operation of electricity over the year', fontsize=17)
        fig.tight_layout()
        plt.savefig("operation.png", bbox_inches="tight", dpi=300)

        return df2, p_by_carrier


    def h2_data(self, bus):
        ely = self.links.loc[self.links[self.links.index.str.contains("electrolyser")].index]
        ely_index = ely[ely['p_nom_opt'] != 0].index.tolist()
        h2stor = self.stores.loc[self.stores[self.stores.index.str.contains("hydrogen storage")].index]
        h2stor_index = h2stor[h2stor['e_nom_opt'] != 0].index.tolist()
        print("RESULTS: Investments in {} MW of electrolysers.".format(ely.p_nom_opt.sum()))
        print("RESULTS: Investments in {} MWh of hydrogen storage ({} kgH2).".format(h2stor.e_nom_opt.sum(),
                                                                                     h2stor.e_nom_opt.sum() * 1000 / 33.33))

        self.stores_t.e[h2stor_index].plot(title="Energy stored in hydrogen storages over the year")
        if bus:
            # # Plot d'une figure qui montre le fonctionnement sur une station (ici Le Gol)
            # df = pd.concat([self.loads_t['p_set']['Le Gol hydrogen buses load'], -self.links_t['p1']['electrolyser Le Gol'], -self.links_t['p1']['compressor Le Gol'], self.stores_t['p']['hydrogen storage hp Le Gol']], axis=1)
            # df = df * 1000 / 33.33
            # df = df.rename(columns={"Le Gol hydrogen buses load": "Demande en hydrogène",
            #                         "electrolyser Le Gol": "Quantité en sortie d'électrolyseur",
            #                         "compressor Le Gol": "Quantité en sortie de compresseur",
            #                         "hydrogen storage hp Le Gol": "Quantité entrante/sortante du stockage"})
            # ax = df[72:72 + 24 * 3].plot()
            # plt.grid()
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
            #           fancybox=True, ncol=1, frameon=False, fontsize=17)
            # ax.set_xlabel(" ", fontsize=17)
            # ax.tick_params(axis='both', which='both', labelsize=14)
            # ax.set_ylabel("Quantité d'hydrogène (kgH2)", fontsize=17)
            # plt.tight_layout()
            # plt.savefig("hydrogen.png", bbox_inches="tight", dpi=300)

            # network.loads_t.p_set['Marquet hydrogen buses load'].sum() + network.loads_t.p_set['Le Gol hydrogen buses load'].sum() + network.loads_t.p_set['Bois Rouge hydrogen buses load'].sum()

            return ely_index, h2stor_index
        else:
            fc = self.links.loc[self.links[self.links.index.str.contains("fuel cell")].index]
            fc_index = fc[fc['p_nom_opt'] != 0].index.tolist()
            print("RESULTS: Investments in {} MW of fuel cell.".format(fc.p_nom_opt.sum()))
            return ely_index, h2stor_index, fc_index
