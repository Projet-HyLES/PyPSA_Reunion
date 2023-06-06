import pypsa
import ast
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from tictoc import tic, toc
from matplotlib import pyplot as plt
from pypsa.plot import add_legend_patches, add_legend_circles, add_legend_lines
import functions_used as functions
import additional_constraints as cs
from electrical_grid import ElectricalGrid, ExistingStorages, AdditionalStorages
from electricity_production import PV, Wind, BaseProduction
from electrical_demand import ElectricalDemand
from hydrogen_elements import H2Chain, H2Demand


# Definition of the whole energy network

class EnergyNetwork(pypsa.Network):

    def __init__(self, snap):
        if snap.size != 8760:
            raise ValueError(
                'ERROR: a period of one year must be simulated.')  # TODO deux options : laisser comme ça ou autoriser en segmentant si moins d'un an, et si plus d'un an ?
        # Adding new attributes to specific components
        override_component_attrs = pypsa.descriptors.Dict(
            {k: v.copy() for k, v in pypsa.components.component_attrs.items()})
        for i in ['Generator', 'Link', 'Store', 'StorageUnit', 'Line']:  # Components involved
            override_component_attrs[i].loc["env_f"] = ["float", "kgCO2eq/MW", 0.0, "fixed environmental impact",
                                                        "Input (optional)"]
            override_component_attrs[i].loc["env_v"] = ["float", "kgCO2eq/MWh", 0.0, "variable environmental impact",
                                                        "Input (optional)"]
            override_component_attrs[i].loc["water_f"] = ["float", "m3/MW", 0.0, "fixed water consumption",
                                                          "Input (optional)"]
            override_component_attrs[i].loc["water_v"] = ["float", "m3/MWh", 0.0, "variable water consumption",
                                                          "Input (optional)"]
        pypsa.Network.__init__(self, override_component_attrs=override_component_attrs)
        self.cons = None
        self.set_snapshots(snap)
        self.scenario = None
        self.data = None
        self.data_path = None
        self.eleclines = None
        self.stations = None
        self.horizon = None
        self.climate_scenario = None
        self.vehicles_scenario = None
        self.buses_scenario = None

    def import_network(self, data_path, h2, h2bus, h2disp, h2size, ext):
        """
        Import and definition of the energy network
        :param data_path: str, path for the data files
        :param h2: str, hydrogen scenario simulated
        :param h2bus: str, hydrogen bus scenario simulated
        :param h2disp: int, number of dispensers for the hydrogen bus scenario simulated
        :param h2size: TODO fonctionnalité pas encore utilisée
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        :return: list of already used and new electricity production technologies
        """
        data, lines = functions.import_from_excel_folder(data_path, self.snapshots.year.unique()[0])
        self.data = data
        self.data_path = data_path
        self.eleclines = lines
        self.scenario = str(
            data['network'][data['network'].index == 'production scenario'].dropna(axis=1).values[0][0])
        self.cons = str(
            data['network'][data['network'].index == 'consumption scenario'].dropna(axis=1).values[0][0])
        self.climate_scenario = str(
            data['network'][data['network'].index == 'climate scenario'].dropna(axis=1).values[0][0])
        self.vehicles_scenario = data['network'][data['network'].index == 'vehicles scenario'].dropna(
            axis=1).values.flatten().tolist()
        buses = data['network'][data['network'].index == 'buses scenario'].dropna(axis=1)
        self.buses_scenario = [buses['List 1'].values[0],
                               ast.literal_eval(buses['List 2'].values[0]),
                               ast.literal_eval(buses['List 3'].values[0])]

        self.horizon = self.snapshots.tolist()

        self.data['meteo_r'] = pd.read_csv(
            self.data_path + "/rayonnement_tmy_" + str(self.snapshots.year.unique().values[0]) + ".csv", sep=',',
            encoding='latin-1',
            index_col=0)
        self.data['meteo_r'].index = self.horizon

        self.data['meteo_t'] = pd.read_csv(
            self.data_path + "/BRIO/T_" + str(self.snapshots.year.unique().values[0])[-2:]
            + '_' + self.climate_scenario + ".csv", sep=',', encoding='latin-1',
            index_col=0)
        self.data['meteo_t'].index = self.horizon

        self.data['wind'] = pd.read_csv(self.data_path + "/data_wind_2019_80m.csv", sep=';', encoding='latin-1',
                                        index_col=0)  # m/s
        self.data['wind'].sort_index(inplace=True)
        self.data['wind'].index = self.horizon

        self.data['rain'] = pd.read_csv(
            self.data_path + '/BRIO/Precipitations/Prec_scena' + self.climate_scenario + '_moy'
            + str(self.snapshots.year.unique().values[0])[-2:] + '.csv')
        if not {'timec', 'lon', 'lat', 'pr_corr'}.issubset(self.data['rain'].columns.values.tolist()):
            raise ValueError('ERROR: rainfall file not formatted.')
        self.data['rain']['timec'] = pd.to_datetime(self.data['rain']['timec'], format='%Y-%m-%d %H:%M:%S')

        # Import of the different energy carriers
        self.import_carriers(self.data["carrier"])

        # Import of the different substations
        ElectricalGrid(data, lines).import_buses(self)
        # Import of the different electrical lines
        ElectricalGrid(data, lines).import_lines(self)
        # Import of the existing batteries
        ExistingStorages(data).import_storages(self)

        # Import of the generators
        self.import_generators(data, ext)
        if not ext:
            hydrau = self.generators[self.generators.index.str.contains("Hydraulique")].index.to_list()
            power_file = self.generators.p_nom[hydrau]
            power_file = power_file.rename(lambda x: x[:-12])
            power_file = power_file.to_frame()
            precs = functions.create_weighted_rainfall(self.data['rain'], power_file, self.data['postes'])
            val_min = (precs * 3.67e-5 + 0.183) * power_file.sum() * 8760  # Values studied for La Réunion
            self.data['generator_data'].loc[self.data['generator_data']['technology'] == 'Hydraulique', 'min_year'] = \
                val_min[0]
            val_max = (precs * 3.67e-5 + 0.373) * power_file.sum() * 8760  # Values studied for La Réunion
            self.data['generator_data'].loc[self.data['generator_data']['technology'] == 'Hydraulique', 'max_year'] = \
                val_max[0]

        # Import of the electrical demand
        ElectricalDemand(self, data["postes"], self.vehicles_scenario, self.buses_scenario).import_demand(self, data,
                                                                                                          self.cons)
        # Import of the additionnal batteries on every substation
        AdditionalStorages(data["storage"], data["postes"]).import_storages(self, data["postes"])

        # Import of the hydrogen technologies according to the scenario passed
        if h2 is None:
            pass
        elif h2 == "stock":
            H2Chain(data, data["postes"].index).import_electrolyser(self, h2size)
            H2Chain(data, data["postes"].index).import_h2_storage_lp(self, h2size)
            H2Chain(data, data["postes"].index).import_fc(self, h2size)
        elif (h2 == "bus") or (h2 == "train") or (h2 == "train+bus"):  # TODO en construction
            h2station = self.data['load_car'][self.scenario]
            # H2Chain(data, sorted(set(chain.from_iterable(h2station.values())))).import_electrolyser(self, h2size)
            # H2Chain(data, sorted(set(chain.from_iterable(h2station.values())))).import_compressor(self)
            # H2Chain(data, sorted(set(chain.from_iterable(h2station.values())))).import_h2_storage_hp(self)
            H2Chain(data, data["load_train"].index).import_electrolyser(self, h2size)
            H2Chain(data, data["load_train"].index).import_compressor(self)
            H2Chain(data, data["load_train"].index).import_h2_storage_hp(self)
            # H2Chain(data, data["load_train"].index).import_expander(self)
            # H2Chain(data, data["load_train"].index).import_h2_storage_lp(self, h2size)
            # H2Chain(data, data["load_train"].index).import_fc(self, h2size)
            if "bus" in h2:  # TODO voir les stations
                H2Demand(h2station, self.data_path).import_h2_bus(self, h2bus, h2disp)
            if "train" in h2:
                H2Demand(data["load_train"].index, self.data_path).import_h2_train(self)

            # add_postes = data["postes"].index
            # for i in range(len(data["load_train"].index)):
            #     add_postes = add_postes.delete(add_postes.get_loc(data["load_train"].index[i]))
            # H2Chain(data, add_postes).import_electrolyser(self, h2size)
            # H2Chain(data, add_postes).import_h2_storage_lp(self, h2size)
            # H2Chain(data, add_postes).import_fc(self, h2size)

        elif (h2 == "stock+bus") or (h2 == "stock+train") or (
                h2 == "stock+bus+train"):  # TODO vérifier la modélisation (détendeur, lp, hp)
            h2station = self.data['load_car'][self.scenario]
            H2Demand(h2station, data_path).import_h2_bus(self, h2bus, h2disp)
            H2Chain(data, h2station).import_electrolyser(self, h2size)
            H2Chain(data, h2station).import_compressor(self)
            H2Chain(data, h2station).import_h2_storage_hp(self)
            H2Chain(data, h2station).import_fc(self, h2size)

        return ast.literal_eval(data['network'].loc['generation base', 'List 1']), \
            ast.literal_eval(data['network'].loc['generation new', 'List 1'])

    def import_carriers(self, data):
        """
        Function to import the different energy carriers involved
        :param data: table of the carriers with their attributes
        :return: None
        """
        self.madd("Carrier", data["name"].tolist(), color=data["color"].tolist())

    def import_generators(self, data, ext):
        """
        Function to import the generators of the energy system
        :param data: big file with all the data
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        :return: None
        """
        generators = data["generator"].sort_values(
            by=["Poste source", "Filière"])  # Values are sorted by station and carrier
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

                if "PV+stockage" in fil:  # TODO probablement devoir régler ce problème de PV+stockage un jour
                    continue

                elif "PV" in fil:
                    PV(data["generator_data"]).import_pv(self, round(total_capa, 2) / 1000, data["meteo_t"][ps], data["meteo_r"][ps], ps,
                                                         ext)  # TODO round parce qu'il y avait un beug avec ext=True à cause de la pbq float

                elif fil == "Eolien":
                    Wind(data["generator_data"], "Vestas").import_wind(self, round(total_capa, 2) / 1000, data["wind"][ps], ps, ext)

                elif fil == "Eolien offshore":  # TODO distinction de modèle à faire
                    Wind(data["generator_data"], "Haliade").import_wind(self, round(total_capa, 2) / 1000, data["wind"][ps], ps, ext)

                else:
                    BaseProduction(data["generator_data"], fil).import_base(self, round(total_capa, 2) / 1000, ps, ext)
                total_capa = 0

    def optimization(self, solver, solver_options, h2, sec_base, sec_new, obj, water, ext):
        """
        Function for the creation of the optimization problem and its solving
        :param solver: str, solver used
        :param solver_options: dict, keyword arguments used by the solver
        :param h2: str, hydrogen scenario simulated
        :param sec_base: list, production sectors already installed (may only work for Reunion)
        :param sec_new: list, production sectors newly installed (may only work for Reunion)
        :param obj: str, type of the optimisation
        :param water: float, limit for water consumption TODO à construire
        :param ext: bool, switch to allow the capacity of some generators to be extendable
        :return: costs and environmental impact TODO ou n'importe quoi d'autre en soi
        """
        print("INFO: creating '{}' optimization...".format(obj))
        model = self.optimize.create_model()
        # model = self.optimize.create_model(transmission_losses=3)  # TODO update PyPSA v0.23.0 pour la suite + voir quel facteur ?

        # Bounds directly on the variables for nominal power  # TODO est-ce que ça a vraiment un impact ? en gros les bornes sur PyPSA ne sont définies qu'en contraintes (surprenant) et là on borne les variables directement (dans l'objectif de gagner du temps de calcul mais c'est pas sûr que ça fonctionne)
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

        # Constraints for the definition of the hydrogen chain
        if h2 == "train":  # TODO voir pour généraliser train et bus
            H2Chain(self.data, self.data["load_train"].index.to_series()).constraint_prodsup_bus(self, model,
                                                                                                 self.horizon)

        # Constraints for the definition of the existing storages
        ExistingStorages(self.data).constraints_existing_battery(self, model, self.horizon)

        # Constraints for the definition of the additional storages
        AdditionalStorages(self.data["storage"], self.data["postes"]).constraints_additionnal_battery(self, model)

        # Constraints for the definition of the disponibility and annual limit of electricity generation technologies
        hydrau = self.generators[self.generators.index.str.contains("Hydraulique")].index.to_list()
        hydrau_xa = pd.Series(hydrau).to_xarray()
        hydrau_xa = hydrau_xa.rename({'index': 'hydrau'})
        BaseProduction(self.data["generator_data"], "Hydraulique").constraint_disp(self, model, self.snapshots, hydrau_xa, ext)
        BaseProduction(self.data["generator_data"], "Hydraulique").constraint_min_max(self, model, self.snapshots, hydrau, ext)

        bioenergie = self.generators[self.generators.index.str.contains("Bioénergie")].index.to_list()
        bioenergie_xa = pd.Series(bioenergie).to_xarray()
        bioenergie_xa = bioenergie_xa.rename({'index': 'bioenergie'})
        BaseProduction(self.data["generator_data"], "Bioénergie").constraint_disp(self, model, self.snapshots, bioenergie_xa, ext)

        bioethanol = self.generators[self.generators.index.str.contains("TAC bioéthanol")].index.to_list()
        bioethanol_xa = pd.Series(bioethanol).to_xarray()
        bioethanol_xa = bioethanol_xa.rename({'index': 'bioethanol'})
        BaseProduction(self.data["generator_data"], "TAC bioéthanol").constraint_disp(self, model, self.snapshots,
                                                                                      bioethanol_xa, ext)
        for i in sec_new:
            data_list = self.generators.index.str.contains(i)
            if data_list.any():
                if i == 'Geothermie':
                    index_list = self.generators[data_list].index.to_list()
                    index_list_xa = pd.Series(index_list).to_xarray()
                    index_list_xa = index_list_xa.rename({'index': i})
                    BaseProduction(self.data["generator_data"], i).constraint_disp(self, model, self.snapshots, index_list_xa, False)
                    BaseProduction(self.data["generator_data"], i).constraint_min_max(self, model, self.snapshots,
                                                                                      index_list, False)
                else:
                    index_list = self.generators[data_list].index.to_list()
                    index_list_xa = pd.Series(index_list).to_xarray()
                    index_list_xa = index_list_xa.rename({'index': i})
                    BaseProduction(self.data["generator_data"], i).constraint_disp(self, model, self.snapshots, index_list_xa, False)
                    BaseProduction(self.data["generator_data"], i).constraint_min_max(self, model, self.snapshots, index_list, False)

        if ext:
            # Only one potential for geothermal energy and OTEC: all or nothing (/!\ MILP /!\)
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

        # TODO Modèle biomasse fait rapidement, à update plus tard
        # if ext:
        #     biomasse = self.generators[self.generators.index.str.contains("asse")].index.to_list() + bioenergie
        #     model.add_constraints(
        #         sum(model.variables["Generator-p"][j, i] for i in biomasse + bioethanol for j in list(self.snapshots)) -
        #         1100000 * sum(model.variables["Generator-p_nom"][i] for i in biomasse) / sum(
        #             self.generators["p_nom_max"][i] for i in biomasse) <= 0,
        #         name="limit2_biomasse")
        biomasse = self.generators[self.generators.index.str.contains("Biomasse")].index.to_list() + bioenergie + bioethanol
        bagasse = self.generators[self.generators.index.str.contains("Bagasse")].index.to_list()
        model.add_constraints(
                 sum(model.variables["Generator-p"][j, i] for i in biomasse+bagasse for j in list(self.snapshots)) <= 1100000,
                 name="limit2_biomasse")
        # model.add_constraints(
        #     sum(model.variables["Generator-p"][j, i] for i in bagasse for j in list(self.snapshots)) <= 1100000,
        #     name="limit2_bagasse")

        # Constraints for water consumption
        # v_water, c_water = cs.impact_constraint(self, model, 'water')
        # model.add_constraints(v_water <= water - c_water, name="water_impact")

        if obj == 'multi':  # TODO à tester (est-ce possible d'optimiser à nouveau sans reconstruire ?), manque front de Pareto et enregistrement de chaque système optimisé
            t = toc(False)
            print("INFO: creating the model took {} seconds.".format(t))
            tic()

            self.optimize.solve_model(solver_name=solver, **solver_options)
            if not self.model.status == 'ok':
                raise ValueError('ERROR: optimization is infeasible, results cannot be plotted.')
            network_cost = self  # TODO tester si possible

            min_costs = cs.impact_result(self, 'cost')
            max_env = cs.impact_result(self, 'env')

            obj_stock = model.objective
            model.objective = cs.impact_constraint(self, model, obj)[0].to_linexpr()
            self.optimize.solve_model(solver_name=solver, **solver_options)
            if not self.model.status == 'ok':
                raise ValueError('ERROR: optimization is infeasible, results cannot be plotted.')
            network_env = self  # TODO tester si possible

            min_env = cs.impact_result(self, 'env')

            step = (max_env - min_env) / 5  # nombre d'itérations fixé arbitrairement
            for i in list(range(min_env, max_env, step)):
                # Constraints for environmental impact within multi-objective optimisation
                v_env, c_env = cs.impact_constraint(self, model, 'env')
                model.add_constraints(v_env <= i - c_env,
                                      name="env_impact")  # à voir pour implémenter la méthode augmentée par la suite
                self.optimize.solve_model(solver_name=solver, **solver_options)
                if not self.model.status == 'ok':
                    raise ValueError('ERROR: optimization is infeasible, results cannot be plotted.')

            return network_cost, network_env


        else:
            if obj == 'env':
                model.objective = cs.impact_constraint(self, model, obj)[0].to_linexpr()

            t = toc(False)
            print("INFO: creating the model took {} seconds.".format(t))
            tic()
            self.optimize.solve_model(solver_name=solver, **solver_options)

            if not self.model.status == 'ok':
                raise ValueError('ERROR: optimization is infeasible, results cannot be plotted.')

            return cs.impact_result(self, 'cost'), cs.impact_result(self, 'env'), cs.impact_result(self, 'water')

    def plot_network(self, status, stor, elec, fc):
        """
        Function for the plot of the network before and after optimisation
        :param status: str, network before of adter optimisation to be plotted
        :param stor: bool, swith to plot the locations and sizes of the storages after optimisation
        :param elec: bool, switch to plot the locations and sizes of the electrolyzers after optimisation
        :param fc: bool, switch to plot the locations and sizes of the fuel cells after optimisation
        :return: None
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
                        self.stores['elec bus'][i] = 'electricity bus ' + i[17:]
                gen = self.stores.groupby(['elec bus', 'carrier']).e_nom_opt.sum()
                lines = self.lines.s_nom_opt / 10
                title = "Reunion's electricity grid after optimization - storages"
                legend1 = self.carriers.loc[self.stores.carrier.unique()]['color']
                legend2 = self.stores.carrier.unique()
                save = "network_map_stor.png"
            elif elec:
                bus_sizes = [50, 100]  # in MW
                unit = 'MW'
                gen = self.links.loc[self.links[self.links.index.str.contains("electrolyser")].index][
                    ['bus0', 'p_nom_opt']].set_index('bus0').squeeze()
                lines = self.lines.s_nom_opt / 10
                title = "Reunion's electricity grid before optimization - electrolysers"
                legend1 = None
                legend2 = None
                save = "network_map_elec.png"
            elif fc:
                bus_sizes = [50, 100]  # in MW
                unit = 'MW'
                gen = self.links.loc[self.links[self.links.index.str.contains("electrolyser")].index][
                    ['bus1', 'p_nom_opt']].set_index('bus1').squeeze()
                lines = self.lines.s_nom_opt / 10
                title = "Reunion's electricity grid before optimization - fuel cells"
                legend1 = None
                legend2 = None
                save = "network_map_fc.png"

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        self.plot(ax=ax, title=title, color_geomap=True, bus_sizes=gen / 3e5,
                  line_widths=lines,
                  branch_components=['Line'],
                  boundaries=[55.1115043336971, 55.97942417175307, -20.843415164623533, -21.424694661983377])
        add_legend_patches(
            ax,
            legend1,
            legend2,
            legend_kw={"bbox_to_anchor": (1, 0), **legend_kwargs, "loc": "lower left"},
        )
        add_legend_circles(
            ax,
            [s / 3e5 for s in bus_sizes],
            [f"{s} " + unit for s in bus_sizes],
            legend_kw=legend_circles_dict,
        )
        add_legend_lines(
            ax,
            [s / 10 for s in line_sizes],
            [f"{s} MVA" for s in line_sizes],
            patch_kw={'color': 'rosybrown'},
            legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
        )
        fig.tight_layout()
        fig.savefig(save, bbox_inches="tight", dpi=300)

    def generator_data(self):
        """
        Function for the plot + informations about the electricity mix after optimisation
        :return: dataframe with hourly intermittent rate
        """
        gen = self.generators_t.p
        pow = self.generators.p_nom_opt
        thermique = []
        charbonbagasse = []
        hydraulique = []
        pv = []
        pvstock = []
        eolien = []
        offshore = []
        bioener = []
        biomasse = []
        bagasse = []
        geothermie = []
        etm = []
        for i in gen.columns:
            if ('TAC fioul/gazole' in i) or ('Moteur Diesel' in i):
                thermique.append(i)
            elif 'Thermique charbon/bagasse' in i:
                charbonbagasse.append(i)
            elif 'Hydraulique' in i:
                hydraulique.append(i)
            elif 'PV+stockage' in i:
                pvstock.append(i)
            elif 'Eolien offshore' in i:
                offshore.append(i)
            elif 'Eolien' in i:
                eolien.append(i)
            elif ('TAC bioéthanol' in i) or ('Bioénergie' in i):
                bioener.append(i)
            elif 'Bagasse' in i:
                bagasse.append(i)
            elif 'Biomasse' in i:
                biomasse.append(i)
            elif 'Geothermie' in i:
                geothermie.append(i)
            elif 'ETM' in i:
                etm.append(i)
            else:
                pv.append(i)

        df = pd.concat([gen[hydraulique].sum(axis=1).rename('Hydraulic'), gen[offshore].sum(axis=1).rename('Offshore'),
                        gen[eolien].sum(axis=1).rename('Wind'), gen[bioener].sum(axis=1).rename('Bioenergy'),
                        gen[bagasse].sum(axis=1).rename('Bagass'), gen[biomasse].sum(axis=1).rename('Biomass'),
                        gen[geothermie].sum(axis=1).rename('Geothermal energy'), gen[etm].sum(axis=1).rename('OTEC'),
                        gen[pv].sum(axis=1).rename('PV')], axis=1)
        fig = plt.figure()
        df.sum().plot.pie(title='Electricity mix over the simulated year', autopct='%1.1f%%')
        fig.tight_layout()
        fig.savefig("electricity_mix.png", bbox_inches="tight", dpi=300)

        # Duration curve  # TODO considérer les stockages dans le graph ? (>0 injection et <0 soutirage)
        df1 = pd.concat([(gen[hydraulique].sum(axis=1) / pow[hydraulique].sum()).rename('Hydraulic').sort_values(
            ascending=False).reset_index(),
                         (gen[offshore].sum(axis=1) / pow[offshore].sum()).rename('Offshore').sort_values(
                             ascending=False).reset_index(),
                         (gen[eolien].sum(axis=1) / pow[eolien].sum()).rename('Wind').sort_values(
                             ascending=False).reset_index(),
                         (gen[bioener].sum(axis=1) / (pow[bioener].sum()+41)).rename('Bioenergy').sort_values(
                             ascending=False).reset_index(),  # pbq : TAC non extendable donc p_nom_opt à 0
                         (gen[bagasse].sum(axis=1) / pow[bagasse].sum()).rename('Bagass').sort_values(
                             ascending=False).reset_index(),
                         (gen[biomasse].sum(axis=1) / pow[biomasse].sum()).rename('Biomass').sort_values(
                             ascending=False).reset_index(),
                         (gen[geothermie].sum(axis=1) / pow[geothermie].sum()).rename('Geothermal energy').sort_values(
                             ascending=False).reset_index(),
                         (gen[etm].sum(axis=1) / pow[etm].sum()).rename('OTEC').sort_values(
                             ascending=False).reset_index(),
                         (gen[pv].sum(axis=1) / pow[pv].sum()).rename('PV').sort_values(ascending=False).reset_index()],
                        axis=1)
        df1 = df1.drop(['snapshot'], axis=1)
        ax = df1.plot()
        ax.grid(True, linestyle='-.', which='both')
        ax.set_title('Duration curve per sector', fontsize=17)
        ax.set_ylabel("Normalised power (MW/MWmax)", fontsize=17)
        ax.set_xlabel("Cumulative hours for the year", fontsize=17)
        ax.tick_params(axis='both', which='both', labelsize=14)
        plt.tight_layout()
        plt.savefig("duration_curve.png", bbox_inches="tight", dpi=300)

        print("RESULTS: {} MWh of hydroelectricity produced.".format(round(gen[hydraulique].sum(axis=1).sum())))
        print("RESULTS: {} MWh of PV produced.".format(round(gen[pv].sum(axis=1).sum())))
        print("RESULTS: {} MWh of onshore wind produced.".format(round(gen[eolien].sum(axis=1).sum())))
        print("RESULTS: {} MWh of offshore wind produced.".format(round(gen[offshore].sum(axis=1).sum())))
        print("RESULTS: {} MWh of biomass (global) produced.".format(round(
            gen[bioener].sum(axis=1).sum() + gen[bagasse].sum(axis=1).sum() + gen[biomasse].sum(axis=1).sum())))
        print("RESULTS: {} MWh of geothermal energie produced.".format(round(gen[geothermie].sum(axis=1).sum())))
        print("RESULTS: {} MWh of ETM produced.".format(round(gen[etm].sum(axis=1).sum())))

        # Operating points
        stor = self.stores_t.p.sum(axis=1)
        stor[stor < 0] = 0
        df2 = pd.concat([df.sum(axis=1) + stor + self.storage_units_t.p_dispatch.sum(axis=1),
                         (gen[offshore].sum(axis=1) + gen[eolien].sum(axis=1) + gen[pv].sum(axis=1)) * 100 / df.sum(
                             axis=1)], axis=1)
        ax = df2.plot(kind='scatter', x=0, y=1)
        ax.grid(True, linestyle='-.', which='both')
        ax.set_title('Diagram of the operating points of the electrical system', fontsize=17)
        ax.set_ylabel("Intermittent renewable energy rate (%)", fontsize=17)
        ax.set_xlabel("Production (including stored energy) (MW)", fontsize=17)
        ax.tick_params(axis='both', which='both', labelsize=14)
        plt.tight_layout()
        plt.savefig("operating_points.png", bbox_inches="tight", dpi=300)

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

        return df2

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
            # df = pd.concat([self.loads_t['p_set']['Le Gol hydrogen load'], -self.links_t['p1']['Le Gol electrolyzer'], -self.links_t['p1']['Le Gol compressor'], self.stores_t['p']['Le Gol H2 storage']], axis=1)
            # df = df * 1000 / 33.33
            # df = df.rename(columns={"Le Gol hydrogen load": "Demande en hydrogène",
            #                         "Le Gol electrolyzer": "Quantité en sortie d'électrolyseur",
            #                         "Le Gol compressor": "Quantité en sorte de compresseur",
            #                         "Le Gol H2 storage": "Quantité entrante/sortante du stockage"})
            # ax = df[72:72 + 24 * 3].plot()
            # plt.grid()
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
            #           fancybox=True, ncol=1, frameon=False, fontsize=17)
            # ax.set_xlabel(" ", fontsize=17)
            # ax.tick_params(axis='both', which='both', labelsize=14)
            # ax.set_ylabel("Quantité d'hydrogène (kgH2)", fontsize=17)
            # plt.tight_layout()
            # plt.savefig("hydrogen.png", bbox_inches="tight", dpi=300)

            return ely_index, h2stor_index
        else:
            fc = self.links.loc[self.links[self.links.index.str.contains("fuel cell")].index]
            fc_index = fc[fc['p_nom_opt'] != 0].index.tolist()
            print("RESULTS: Investments in {} MW of fuel cell.".format(fc.p_nom_opt.sum()))
            return ely_index, h2stor_index, fc_index
