The code in this repository enables running simulations of La Réunion's high voltage grid based on scenarios for 2030 and 2050, with a focus on exploring the potential roles of hydrogen.
It was created as part of the HyLES project ANR-20-CE05-0035 funded by the French research agency (ANR). More information on the project: https://projects.femto-st.fr/hyles/.
This work is the result of a collaboration between ENERGY-lab at Université de La Réunion and FEMTO-ST at Université Marie et Louis Pasteur. 
The main author of this work is Dr. Agnès François. More details in her Ph.D. thesis: https://theses.fr/2024UBFCA011.

# Requirements 
- PyPSA (`conda install -c conda-forge pypsa`)
- cartopy (`conda install conda-forge::cartopy`)
- openpyxl (`conda install -c anaconda onpenpyxl`)
- *optionnal* WindPowerLib (`pip install windpowerlib`)
- *for Gurobi solver* gurobi and gurobipy (`conda install -c gurobi gurobi` and `pip install gurobipy`) + full license

# Code
File to run for modeling and optimising the system: **run.py**.
All the data used for this modeling can be updated from the Data folder.

This file calls **functions_used.py** where few functions are defined, mostly for the formatting of the data.

In **energy_network.py**, the class EnergyNetwork is defined, importing and optimising the network. In **electricity_production.py**, **electrical_grid.py**, **electrical_demand.py** and **hydrogen_elements.py**, the elements of the network are created following the name of the file.
In **additional_constraints.py**, two functions for the definition of constraints and resulting impacts are introduced.

# Scenarios
A lot of scenarios are introduced in the code and can be found on the first sheet of **data20X0.xlsx**. 
- electricity production: variation in installed electricity generation capacity;
- electricity consumption: variation in basic electricity consumption;
- electric vehicles: percentages of controllable electric fleet, total additional demand for vehicles;
- buses: electric bus frequency scenario;
- climate: BRIO project scenario (IPCC, data not on the repository);
- hydrogen: use of hydrogen in the system;
- multiyear; seeveral years are being simulated

Within the hydrogen scenarios, several scenarios/data are imported depending on the case study:
- scenarios: storage, bus (number of stations and dispensers), train;
- consumption: time series of consumption at HRS according to station configuration.

# Data
The different data required are stored in **.csv** or **.xslx** files. Some data were constructed, and the methodology used can be found in the following reference : [*coming*].


The following data files are used in the code :
- *data20X0.xlsx* :  excel with all data other than time series (technical, economic data, etc.)
    - Carrier : energy vectors or energies used, used only for graphics
    - Generator : technical and economic data for electricity production
    - Rayonnement_ps : links a weather station with radiation data to a source station
    - Température_ps : links a weather station with temperature data to a source station
    - Load_ps : assimilates source substations without transformers to the nearest source substation with a transformer
    - Load_buses : assimilates charging source stations according to bus network and power generation scenario
    - Link : technical and economic data for hydrogen systems
    - Batteries : technical and economic data for existing batteries
    - Storage : technical and economic data for additional storage (batteries, hydrogen)
    - Units
- *postes-sources.csv*: substation data for the power grid. Must contain at least the following information: name, coordinates, presence of transformer, voltage.
- *registre-des-installations-de-production-et-de-stockage.csv*: data on electricity generation facilities. Must contain at least the following information: substation, type, power in kW.
- *htb_souter.csv* and *htb_aer.csv*: grid power line data. Must contain at least the following information: line name, length in km, permissible capacity in MVA.
- *T_30_1.csv*: time series of temperature at source stations according to target horizon and chosen climate change scenario.
- *wind_data.csv*: wind speed time series.
- *Prec_scena_2_moy50.csv*: precipitation time series by latitude/longitude. Must contain at least the following columns: timec, lon, lat, pr_corr.
- *VE-80pilotable-results-1050.csv*: time series of electric vehicle consumption at substations according to fleet management and total electricity demand.
- *VE-40pilotable.xlsx*: Excel with information for creating electric vehicle consumption.
    - Load curve: hourly power according to scenario.
    - Demographic distribution: demographic distribution according to the communes in the case study.
    - Stations: allocation of communes to island substations.
- *conso_bus_urbains.csv*: time series of urban bus power consumption, broken down by bus network and frequency scenario.
- *profil_bus_elec.xlsx*: hourly profile of electric bus load distribution. Divided into two columns, "week" and "Sunday".
- *20X0_Y.csv*: time series of base electricity consumption at the substation, according to electricity consumption scenario. X stands for the year simulated, Y for the climate scenario simulated.
- *Vestas_X.csv* : capacity factor of Vestas wind turbines. 1 is for onshore wind, 2 is for offshore wind.

These complementary data are required to run pypsa-reunion and should be downloaded : https://zenodo.org/records/17149940?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY3YWI1OGZjLTEwMTUtNDgwMS05ZTlkLTY0MzQzMTUyZTIzOSIsImRhdGEiOnt9LCJyYW5kb20iOiIzODJlNDRlNmVjZGNkZjJkYTMxODViYjljYTE1MDNmZCJ9.ZC8xpnOw8MvV7nuCof1MQB3gCptgYxLpAcuCNTzzGjY6dK03IGRIZbij5dLs8yAwogSpzhLdMcMyy_SFvO-BZw

# Remarks
Climate data are not all available on the repository. Indeed, a lot of the data used come from the BRIO¹ project (*Building Resilience in the Indian Ocean*). If you need more data, feel free to contact A. François: *agnes.francois@yahoo.fr*.

<small>1: Marie-Dominique Leroux, François Bonnardot, Stephason François Kotomangazafy, Abdoul-Oikil Said, Philippe Veerabadren, et al.. Régionalisation du changement climatique et développement de services climatiques dans le sud-ouest de l'océan Indien et ses territoires insulaires. METEO FRANCE. 2023. ⟨hal-03966983v7⟩</small>
