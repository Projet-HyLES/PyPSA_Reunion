# Requirements 
- PyPSA (`conda install -c conda-forge pypsa`)
- gurobi and gurobipy (`conda install -c gurobi gurobi` and `pip install gurobipy`) + full license
- openpyxl (`conda install -c anaconda onpenpyxl`)

# Code
File to run for modeling and optimising the system: **run.py**.
This file currently models an electricity scenario of 2050. All the data used for this modeling can be updated from the Data folder.

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
- hydrogen: use of hydrogen in the system.

Within the hydrogen scenarios, several scenarios/data are imported depending on the case study:
- scenarios: storage, bus (number of stations and dispensers), train;
- consumption: time series of consumption at HRS according to station configuration.

# Data
The different data required are stored in **.csv** or **.xslx** files.
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
- *rayonnement_tmy_2050.csv*: time series of TMY radiation at substations for the target timeframe.
- *T_30_1.csv*: time series of temperature at source stations according to target horizon and chosen climate change scenario.
- *data_wind_2019_80m.csv*: wind speed time series (to be updated as soon as possible).
- *Prec_scena_2_moy50.csv*: precipitation time series by latitude/longitude. Must contain at least the following columns: timec, lon, lat, pr_corr.
- *VE-80pilotable-results-1050.csv*: time series of electric vehicle consumption at substations according to fleet management and total electricity demand.
- *VE-40pilotable.xlsx*: Excel with information for creating electric vehicle consumption.
	- Load curve: hourly power according to scenario.
	- Demographic distribution: demographic distribution according to the communes in the case study.
  	- Stations: allocation of communes to island substations.
- *conso_bus_urbains.csv*: time series of urban bus power consumption, broken down by bus network and frequency scenario.
- *profil_bus_elec.xlsx*: hourly profile of electric bus load distribution. Divided into two columns, "week" and "Sunday".
- *Abondance_EE.csv*: time series of basic electricity consumption at the substation, according to electricity consumption scenario.


