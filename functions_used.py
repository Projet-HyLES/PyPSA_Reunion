import numpy as np
import pandas as pd
import math
from shapely.geometry import LineString
from shapely.ops import transform
import pyproj
from pyproj import CRS


def extraction_localisation(data, column):
    """
    Extraction of the longitude and the latitude from a str of the type "latitude,longitude"
    :param data: dataframe where the initial str is and where the final longitude and latitude will be stored
    :param column: column of the dataframe where the str is stored
    :return: the initial dataframe with two additional columns, for latitude and longitude
    """
    lat = np.zeros(len(data))
    long = np.zeros(len(data))
    for i in range(len(data)):
        coord = data[column][i].split(",")
        lat[i] = coord[0]
        long[i] = coord[1]
    data["Lat"] = lat
    data["Long"] = long
    return data


def import_from_excel_folder(path, year):
    """
    Import of the required data from different files
    :param year: simulation year
    :param path: path for the file where data are stored
    :return: dict with all the data compiled
    """
    print("INFO: importing energy network data...")
    data = pd.ExcelFile(path+"/data"+str(year)+".xlsx")
    ps = pd.read_csv(path+"/postes-sources.csv", sep=';', encoding='latin-1', index_col=1)
    if not {"Point Geo", "Transfo", "Voltage"}.issubset(ps.columns.values.tolist()):
        raise ValueError('ERROR: substation file not formatted.')
    extraction_localisation(ps, 'Point Geo')
    components = {
        "network": data.parse("network", index_col=0),
        "postes": ps,
        "carrier": data.parse("carrier"),
        "generator_data": data.parse("generator"),
        "rayonnement_ps": data.parse("rayonnement_ps"),
        "temperature_ps": data.parse("temperature_ps"),
        "load_ps": data.parse("load_ps"),
        "load_buses": data.parse("load_buses"),
        "load_car": data.parse("load_car"),
        "load_train": data.parse("load_train", index_col=0),
        "link": data.parse("link"),
        "batteries": data.parse("batteries", index_col=0),
        "storage": data.parse("storage"),
    }

    scena_prod = str(
        components['network'][components['network'].index == 'production scenario'].dropna(axis=1).values[0][0])

    prod = pd.read_csv(path+"/power_facilities-" + str(year) + "-" + str(scena_prod) + ".csv", sep=';', encoding='latin-1')
    if not {"Poste source", "Filière", "Puissance installée (kW)"}.issubset(prod.columns.values.tolist()):
        raise ValueError('ERROR: generator file not formatted.')
    components["generator"] = prod

    lines_souter = formate_lines(path+"/htb_souter.csv")  # Recovery of underground lines data
    lines_aer = formate_lines(path+"/htb_aer.csv")  # Recovery of overhead lines data
    if not {"Nom de la ligne", "Longueur (km)",
            "Capacite (MVA)"}.issubset(lines_souter.columns.values.tolist()) or not {"Nom de la ligne", "Longueur (km)",
                                                                                     "Capacite (MVA)"}.issubset(lines_aer.columns.values.tolist()):
        raise ValueError('ERROR: line file not formatted.')
    lines = pd.concat([lines_aer, lines_souter])
    lines = lines.fillna('')

    print("INFO: data from files data.xlsx, postes-sources.csv, {}, htb_souter.csv and htb_aer.csv imported."
          .format("registre-des-installations-de-production-et-de-stockage-" + str(year) + "-" + str(scena_prod) + ".csv"))
    return components, lines


def formate_lines(file):
    """
    Formatting of the lines' data.

    :param file: str, CSV file path where the lines data is stored.
    :return: pandas.DataFrame, Dataframe with formatted data.
    """

    # Load CSV file, drop NaN values, and reset index
    lignes = pd.read_csv(file, sep=',', encoding='latin-1').dropna().reset_index()

    # Count the maximum number of "/" in "Nom de la ligne" column to determine the number of "bus" columns
    nb_bus = max(lignes["Nom de la ligne"].str.count('/') + 1)

    # Create a list of column names for "bus" columns
    bus_columns = [f"bus{i}" for i in range(nb_bus)]

    # Split "Nom de la ligne" column into separate "bus" columns
    lignes[bus_columns] = lignes["Nom de la ligne"].str.split('/', expand=True)

    return lignes


def calculate_marginal_costs(fuel_cost, variable_OM, efficiency):
    """
    Function to calculate the equivalent marginal costs
    From https://github.com/PyPSA/pypsa-eur/blob/master/scripts/add_electricity.py
    :param fuel_cost: fuel costs (€/MWh)
    :param variable_OM: variable operation and maintenance costs (€/MWh)
    :param efficiency: efficiency of the technology
    :return: float for marginal costs
    """
    return fuel_cost / efficiency + variable_OM


def calculate_capital_costs(d_r, lifetime, fixed_OM_p, fixed_OM_t, CAPEX, Nyears):
    """
    Function to calculate the equivalent capital costs
    From https://github.com/PyPSA/pypsa-eur/blob/master/scripts/add_electricity.py
    :param d_r: discount rate
    :param lifetime: lifetime (years)
    :param fixed_OM_t: fixed operation and maintenance costs (€/year)
    :param fixed_OM_p: fixed operation and maintenance costs (%/year)
    :param CAPEX: investments (€/MW)
    :param Nyears: number of years simulated
    :return: float for capital costs
    """
    annuity = d_r / (1 - (1 + d_r) ** (- lifetime))
    return (annuity + fixed_OM_p/100) * CAPEX * Nyears + fixed_OM_t * Nyears


def prod_vestas(type, umin, umax, unom, rho, diam, u, capa, x):
    """
    Function to calculate the energy produced by a Vestas wind turbine at a time t.
    :param type: model used (offshore or onshore)
    :param umin: minimal wind speed
    :param umax: maximal wind speed
    :param unom: nominal wind speed
    :param rho: air density
    :param diam: rotor swept area exposed
    :param u: wind speed
    :param capa: capacity of the wind turbine
    :param x: number of turbines
    :return: energy produced
    """
    if (u <= umin) or (u >= umax):
        return 0
    if unom <= u <= umax:
        return capa * x  # in MW
    else:
        if type == "onshore":
            return 1/2 * (-0.01 * u**2 + 0.1324 * u + 0.0177) * rho * u**3 * math.pi * (diam/2)**2 * x/1e6  # in MW
        elif type == "offshore":
            return 1/2 * (-0.0084 * u**2 + 0.1452 * u - 0.147) * rho * u**3 * math.pi * (diam/2)**2 * x/1e6  # in MW


def create_weighted_rainfall(prec_file, power_file, ps):
    """
    Function to create a weighted annual rainfall from rainfall data for the whole island and power capacities installed on the
    different substations.
    :param prec_file: rainfall file, with annual rainfall on different places of the island
    :param power_file: file with the different hydroelectric capacities installed on the different substations
    :param ps: substation file
    :return: annual weighted rainfall for the island
    """
    ps['Lat'] = round(ps['Lat'], 2)
    lat_ps = ps['Lat'].unique()
    lat_ps.sort()
    lat_ps = pd.DataFrame(lat_ps)
    lat_ps['coord ref'] = np.nan

    ps['Long'] = round(ps['Long'], 2)
    long_ps = ps['Long'].unique()
    long_ps.sort()
    long_ps = pd.DataFrame(long_ps)
    long_ps['coord ref'] = np.nan

    long_mf = prec_file['lon'].unique()
    long_mf = pd.DataFrame(long_mf)
    lat_mf = prec_file['lat'].unique()
    lat_mf = pd.DataFrame(lat_mf)

    somme = 0
    for i in lat_ps.index:
        lat_ps.loc[i, 'coord ref'] = lat_mf.iloc[lat_mf.sub(lat_ps.loc[i, 0]).abs().idxmin()].values[0][0]

    for i in long_ps.index:
        long_ps.loc[i, 'coord ref'] = long_mf.iloc[long_mf.sub(long_ps.loc[i, 0]).abs().idxmin()].values[0][0]

    for j in power_file.index:
        lat = ps["Lat"].loc[ps.index == j]  # latitude of the substation
        long = ps["Long"].loc[ps.index == j]  # longitude of the substation
        prec_ok = prec_file[(prec_file['lon'] == long_ps["coord ref"].loc[long_ps[0] == long.values[0]].values[0]) & (
                    prec_file['lat'] == lat_ps["coord ref"].loc[lat_ps[0] == lat.values[0]].values[0])]['pr_corr']
        prec_ok = prec_ok.values[0]
        somme += (prec_ok * power_file.loc[j] / power_file.sum()).values[0]

    return somme


def calculate_distance(pointA, pointB):
    """
    Function to calculate the distance between two points pointA and pointB
    """
    ligne = LineString([pointA, pointB])
    crs_4326 = CRS("EPSG:4326")
    crs_proj = CRS("EPSG:3727")
    project = pyproj.Transformer.from_crs(crs_4326, crs_proj).transform
    return transform(project, ligne).length

