import numpy as np
import pandas as pd
import datetime
import os.path
import holidays
import pytz
import math
import sys


def extraction_localisation(data, column):
    """
    Extraction of the longitude and the latitude from a str of the type(latitude,longitude)
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


def import_from_excel_folder_old(file_data, file_ps, file_prod):
    """
    Import of the required data from different files.
    **Fonction obsolète, utilisée avant mon changmenent d'approche du code (départ à La Réunion)**
    :param file_data: Excel file where the parameters of the system are stored
    :param file_ps: CSV file with source substations data
    :param file_prod: CSV file with electricity generators data
    :return: dict with all the data compiled
    """
    print("Defining the energy network from the files: {}, {} and {}".format(file_data, file_ps, file_prod))
    data = pd.ExcelFile(file_data)
    ps = pd.read_csv(file_ps, sep=';', encoding='latin-1')
    extraction_localisation(ps, 'Point Geo')
    prod = pd.read_csv(file_prod, sep=';', encoding='latin-1')
    components = {
        "network": data.parse("network"),
        "bus": ps,
        "carrier": data.parse("carrier"),
        "generator_data": data.parse("generator"),
        "generator": prod,
        "rayonnement_ps": data.parse("rayonnement_ps"),
        "temperature_ps": data.parse("temperature_ps"),
        "load_ps": data.parse("load_ps"),
        "load_buses": data.parse("load_buses"),
        "link": data.parse("link"),
        "batteries": data.parse("batteries"),
        "storage": data.parse("storage"),
    }

    print("Data from files {}, {} and {} imported.".format(file_data, file_ps, file_prod))
    return components


def import_from_excel_folder(path, year):
    """
    Import of the required data from different files
    :param year: simulation year
    :param path:
    :return: dict with all the data compilated
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

    prod = pd.read_csv(path+"/registre-des-installations-de-production-et-de-stockage-" + str(year) + "-" + str(scena_prod) + ".csv", sep=';', encoding='latin-1')
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
          .format("registre-des-installations-de-production-et-de-stockage-2050-" + str(scena_prod) + ".csv"))
    return components, lines


def formate_lines(file):
    """
    Formatting of the lines data.

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


def formate_loads(i, type):
    """
    Formatting of the demand data
    :param i: name of the source substation from which demand data need formatting
    :param type: str, week or weekend
    :return: Dataframe with week and weekend type data of residential and tertiary consumptions
    """
    conso = pd.read_csv("conso/" + i + ".csv", sep=',', encoding='latin-1')

    conso_tertiaire = conso.loc[conso["Type"] == type].loc[conso["Secteur"] == "Tertiaire"].copy()
    conso_tertiaire = conso_tertiaire.reset_index()
    for h in conso_tertiaire.index:
        conso_tertiaire["Heure"].iloc[h] = (
            datetime.datetime.strptime(conso_tertiaire["Heure"][h], '%H:%M').time()).hour

    conso_residentiel = conso.loc[conso["Type"] == type].loc[conso["Secteur"] == "Résidentiel"].copy()
    conso_residentiel = conso_residentiel.reset_index()
    for h in conso_residentiel.index:
        conso_residentiel["Heure"].iloc[h] = (
            datetime.datetime.strptime(conso_residentiel["Heure"][h], '%H:%M').time()).hour

    return conso_tertiaire, conso_residentiel


def formate_meteo_tmy(data, folder, horizon, ps):
    """
    Formatting of the irradiation and temperature data for the calculation of the hourly PV production. Use of TMY data.
    :param data: dictionnary with data of the dependance of the stations
    :param folder: folder where the meteo files are stored
    :param horizon: horizon of the modeling
    :param ps: list of the source substations where there are PV installations
    :return: dictionnary updated with the formatted meteo data
    """
    data["meteo_r"] = pd.DataFrame(index=horizon, columns=ps)
    data["meteo_t"] = pd.DataFrame(index=horizon, columns=ps)
    for i in ps:
        rayonnement_bool = data["rayonnement_ps"].isin([i]).any()  # meteo station corresponding to the source substation
        temperature_bool = data["temperature_ps"].isin([i]).any()
        ref_rayonnement = rayonnement_bool[rayonnement_bool == True].index[0]
        ref_temperature = temperature_bool[temperature_bool == True].index[0]

        rayonnement = pd.read_csv(os.path.join(folder, 'Rayonnement', ref_rayonnement + '.csv'), sep=',', encoding='latin-1')
        temperature = pd.read_csv(os.path.join(folder, 'Temperature', ref_temperature + '.csv'), sep=',', encoding='latin-1')
        rayonnement['Date'] = pd.to_datetime(rayonnement['Date'], format='%Y-%m-%d')
        temperature['Date'] = pd.to_datetime(temperature['Date'], format='%Y-%m-%d')
        if type(rayonnement['Heure'].iloc[0]) == str:  # hours data are not all in the same format : H and H:M:S. This condition is for the second format
            rayonnement['Heure'] = pd.to_datetime(rayonnement['Heure'], format='%H:%M:%S').dt.strftime("%H")
            rayonnement['Heure'] = pd.to_numeric(rayonnement['Heure'])

        temperature['Heure'] = pd.to_datetime(temperature['Heure'], format='%H:%M:%S').dt.strftime("%H")  # hours all in the same format
        temperature['Heure'] = pd.to_numeric(temperature['Heure'])

        for j in data["meteo_r"].index:
            data["meteo_r"][i].loc[j] = rayonnement.loc[(rayonnement['Date'].dt.month == j.month) & (rayonnement['Date'].dt.day == j.day) & (rayonnement["Heure"] == j.hour), "GLO"].values * 2.78
            data["meteo_t"][i].loc[j] = temperature.loc[(temperature['Date'].dt.month == j.month) & (temperature['Date'].dt.day == j.day) & (temperature["Heure"] == j.hour), "TempÃ©rature"].values

            if len(data["meteo_r"][i].loc[j]) == 0:  # some data are missing
                data["meteo_r"][i].loc[j] = np.NaN  # first to NaN, then NaN are filled with the mean of the irradiation

            if len(data["meteo_t"][i].loc[j]) == 0:
                data["meteo_t"][i].loc[j] = np.NaN

    # NaN created for missing values are filled with the average of the values for the other stations, at the same time
    data["meteo_r"] = data["meteo_r"].T.fillna(data["meteo_r"].mean(axis=1)).T
    data["meteo_t"] = data["meteo_t"].T.fillna(data["meteo_t"].mean(axis=1)).T

    return data


def formate_meteo(data, files, horizon, ps):
    """
    Formatting of the irradiation and temperature data for the calculation of the hourly PV production. Use of the real data. Files aren't build but used from scratch.
    :param data: dictionnary with data of the dependance of the stations
    :param files: folder where the meteo files are stored
    :param horizon: horizon of the modeling
    :param ps: list of the source substations where there are PV installations
    :return: dictionnary updated with the formatted meteo data
    """
    rayonnement = files[0]
    rayonnement['DATE'] = pd.to_datetime(rayonnement['DATE'], errors='coerce')
    rayonnement.set_index('DATE', inplace=True)
    rayonnement = rayonnement.tz_localize(pytz.timezone('UTC'))
    rayonnement.index = rayonnement.index.tz_convert(None)

    temperature = files[1]
    temperature['_time'] = pd.to_datetime(temperature['_time'], errors='coerce')
    temperature.set_index('_time', inplace=True)
    temperature = temperature.tz_localize(pytz.timezone('UTC'))
    temperature.index = temperature.index.tz_convert(None)

    temperature_denis = files[2]
    temperature_denis['_time'] = pd.to_datetime(temperature_denis['_time'], errors='coerce')
    temperature_denis.set_index('_time', inplace=True)
    temperature_denis = temperature_denis.tz_localize(pytz.timezone('UTC'))
    temperature_denis.index = temperature_denis.index.tz_convert(None)

    rayonnement_ok = rayonnement.iloc[rayonnement.index.isin(horizon)]
    temperatures_ok = temperature.iloc[temperature.index.isin(horizon)]
    temperatures_den_ok = temperature_denis.iloc[temperature_denis.index.isin(horizon)]
    rayonnement = pd.DataFrame(columns=ps)
    temperature = pd.DataFrame(columns=ps)

    for i in ps:
        rayonnement_bool = data["rayonnement_ps"].isin([i]).any()  # meteo station corresponding to the source substation
        temperature_bool = data["temperature_ps"].isin([i]).any()
        ref_rayonnement = rayonnement_bool[rayonnement_bool == True].index[0]
        ref_temperature = temperature_bool[temperature_bool == True].index[0]

        rayonnement[i] = rayonnement_ok.loc[rayonnement_ok['NOM'] == ref_rayonnement, 'GLO'] * 2.78
        if ref_temperature == 'GILLOT-AEROPORT':  # Saint Denis a son propre fichier de températures
            temperature[i] = temperatures_den_ok['T_GILLOT-AEROPORT']

        else:
            temperature[i] = temperatures_ok.loc[temperatures_ok['station_name'] == ref_temperature, 'T']

    # NaN created for missing values are filled with the average of the values for the other stations, at the same time
    rayonnement = rayonnement.T.fillna(rayonnement.mean(axis=1)).T
    temperature = temperature.T.fillna(temperature.mean(axis=1)).T

    return rayonnement, temperature


def calculate_marginal_costs(fuel_cost, variable_OM, efficiency):
    """
    The function calculates the equivalent marginal and capital costs
    Formule taken from https://github.com/PyPSA/pypsa-eur/blob/master/scripts/add_electricity.py
    :param fuel_cost:
    :param variable_OM:
    :param efficiency:
    :return: float for marginal and capital costs
    """
    return fuel_cost / efficiency + variable_OM


def calculate_capital_costs(d_r, lifetime, fixed_OM_p, fixed_OM_t, CAPEX, Nyears):
    """
    The function calculates the equivalent marginal and capital costs
    Formule taken from https://github.com/PyPSA/pypsa-eur/blob/master/scripts/add_electricity.py
    :param d_r: discount rate
    :param lifetime: lifetime (years)
    :param fixed_OM_t: fixed operation and maintenance costs (€/year)
    :param fixed_OM_p: fixed operation and maintenance costs (%/year)
    :param CAPEX: investments (€/MW)
    :param Nyears: years simulated
    :return: float for marginal and capital costs
    """
    annuity = (1 - (1 + d_r) ** (- lifetime)) / d_r
    return (annuity + fixed_OM_p/100) * CAPEX * Nyears + fixed_OM_t * Nyears


def get_holidays(days):
    """
    This function recover the week/weekend type of given days of a specific year. Public holidays are taken for weekend days.
    :param days: DatetimeIndex which must be differenciated
    :return: DataFrame with days as index and for each day its type (week or weekend)
    """
    df_holidays = pd.DataFrame(index=days, columns=["Type"])
    fr_holidays = []
    for i in holidays.FR(years=days.year.unique().values).items():
        fr_holidays.append(str(i[0]))
    for date in days:
        if str(date).split()[0] in fr_holidays:
            df_holidays.loc[date] = "holiday"
        elif date.dayofweek == 5:
            df_holidays.loc[date] = "saturday"
        elif date.dayofweek == 6:
            df_holidays.loc[date] = "sunday"
        else:
            df_holidays.loc[date] = "weekday"
    return df_holidays


def prod_vestas(umin, umax, unom, rho, diam, u, capa, x):
    # TODO proposer un modèle différent pour offshore
    """
    Calculate the energy produced by a wind turbine at a time t.
    :param umin: minimal wind speed
    :param umax: maximal wind speed
    :param unom: nominal wind speed
    :param rho: air density
    :param diam: rotor swept area exposed to the wind
    :param u: wind speed
    :param capa: capacity of the wind turbine
    :param x: number of turbines
    :return: the energy produced
    """
    if (u <= umin) or (u >= umax):
        return 0
    if unom <= u <= umax:
        return capa * 1000
    else:
        return 1/2 * (-0.01 * u**2 + 0.1324 * u + 0.0177) * rho * u**3 * math.pi * (diam/2)**2 * x/1000


def creation_profil_bus(horizon, days, cons_week, cons_sunday, data):
    """
    Function to create a year profile for the consumption of buse. From a daily profile (week day or sunday), the function
    produces.
    :param horizon: horizon of the simulated year
    :param days: days of the simulated year
    :param cons_week: total consumption of a week day
    :param cons_sunday: total consumption of a holiday
    :param data: daily load profile (percentages of the fleet loading)
    :return: year profile
    """
    df_holidays = get_holidays(days)
    week = cons_week * data[0] / 100
    sunday = cons_sunday * data[1].squeeze() / data[1].squeeze().sum()
    empty = np.zeros((days.size, 24))
    for h in range(empty.shape[0]):
        if df_holidays.iloc[h].values == "sunday" or df_holidays.iloc[h].values == "holiday":
            if h == empty.shape[0]-1 or df_holidays.iloc[h+1].values == "sunday" or df_holidays.iloc[h+1].values == "holiday":
                empty[h, :] = sunday
            else:
                empty[h, :] = sunday.add(week[12:], fill_value=0)
        else:
            if h == empty.shape[0]-1 or df_holidays.iloc[h + 1].values == "sunday" or df_holidays.iloc[h + 1].values == "holiday":
                empty[h, :] = pd.Series(0, index=week.index).add(week[:12], fill_value=0)
            else:
                empty[h, :] = week
    empty = empty.reshape((-1, 1))
    my_list = map(lambda x: x[0], empty)
    ser = pd.Series(my_list)
    ser.index = horizon
    return ser

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
        lat = ps["Lat"].loc[ps.index == j]  # latitude du poste source
        long = ps["Long"].loc[ps.index == j]  # longitude du poste source
        prec_ok = prec_file[(prec_file['lon'] == long_ps["coord ref"].loc[long_ps[0] == long.values[0]].values[0]) & (
                    prec_file['lat'] == lat_ps["coord ref"].loc[lat_ps[0] == lat.values[0]].values[0])]['pr_corr']
        prec_ok = prec_ok.values[0]
        somme += (prec_ok * power_file.loc[j] / power_file.sum()).values[0]

    return somme
