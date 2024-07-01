
def impact_constraint(n, m, keyword):
    """
    Constraint for the calculation of the environmental impacts
    :param n: network
    :param m: model
    :param keyword: str, impact required
    :return: variable and constant part of the impact
    """
    v = 0
    c = 0
    for i in n.generators.index.tolist():
        if n.generators['p_nom_extendable'][i]:
            v += m.variables['Generator-p_nom'][i] * n.generators[keyword + "_f"][i] + sum(
                m.variables['Generator-p'][t, i] for t in list(n.snapshots)) * n.generators[keyword + "_v"][i]
        else:
            v += sum(m.variables['Generator-p'][t, i] for t in list(n.snapshots)) * n.generators[keyword + "_v"][i]
            c += n.generators['p_nom'][i] * n.generators[keyword + "_f"][i]
    for i in n.links.index.tolist():
        if n.links['p_nom_extendable'][i]:
            v += m.variables['Link-p_nom'][i] * n.links[keyword + "_f"][i] + sum(
                m.variables['Link-p'][t, i] for t in list(n.snapshots)) * n.links[keyword + "_v"][i]
        else:
            v += sum(m.variables['Link-p'][t, i] for t in list(n.snapshots)) * n.links[keyword + "_v"][i]
            c += n.links['p_nom'][i] * n.links[keyword + "_f"][i]
    for i in n.stores.index.tolist():
        if n.stores['e_nom_extendable'][i]:
            v += m.variables['Store-e_nom'][i] * n.stores[keyword + "_f"][i] + sum(
                m.variables['Store-p'][t, i] for t in list(n.snapshots)) * n.stores[keyword + "_v"][i]
        else:
            v += sum(m.variables['Store-p'][t, i] for t in list(n.snapshots)) * n.stores[keyword + "_v"][i]
            c += n.stores['e_nom'][i] * n.stores[keyword + "_f"][i]
    for i in n.storage_units.index.tolist():
        v += sum(
            m.variables['StorageUnit-p_dispatch'][t, i] for t in list(n.snapshots)) * n.storage_units[keyword + "_v"][i]
        c += n.storage_units['p_nom'][i] * n.storage_units[keyword + "_f"][i]
    for i in n.lines.index.tolist():
        v += m.variables['Line-s_nom'][i] * n.lines[keyword + "_f"][i]
    return v, c


def impact_result(n, keyword):
    """
    Calculation of the different impacts
    :param n: network
    :param keyword: str, impact required
    :return: sum over the entire impact
    """
    s = 0
    if keyword == 'cost':
        for i in n.generators.index.tolist():
            s += (n.generators['p_nom_opt'][i] - n.generators['p_nom'][i]) * n.generators["capital_cost"][i] + sum(
                n.generators_t['p'][i][t] for t in list(n.snapshots)) * n.generators["marginal_cost"][i] + n.generators["base_CAPEX"][i]
        for i in n.links.index.tolist():
            s += (n.links['p_nom_opt'][i] - n.links['p_nom'][i]) * n.links["capital_cost"][i] + sum(
                n.links_t['p0'][i][t] for t in list(n.snapshots)) * n.links["marginal_cost"][i]
        for i in n.stores.index.tolist():
            s += (n.stores['e_nom_opt'][i] - n.stores['e_nom'][i]) * n.stores["capital_cost"][i] + sum(
                n.stores_t['p'][i][t] for t in list(n.snapshots)) * n.stores["marginal_cost"][i]
        for i in n.storage_units.index.tolist():
            s += (n.storage_units['p_nom_opt'][i] - n.storage_units['p_nom'][i]) * n.storage_units["capital_cost"][i] + sum(
                n.storage_units_t['p_dispatch'][i][t] for t in list(n.snapshots)) * n.storage_units["marginal_cost"][i]
        for i in n.lines.index.tolist():
            s += (((n.lines['s_nom_opt'][i] < 39) & (n.lines['s_nom_opt'][i] > 26.2)) * (4200 + 25000) * n.lines['length'][i] * 1.2 +
                  ((n.lines['s_nom_opt'][i] < 50) & (n.lines['s_nom_opt'][i] > 44.7)) * (7400 + 25000) * n.lines['length'][i] * 1.2 +
                  ((n.lines['s_nom_opt'][i] < 67) & (n.lines['s_nom_opt'][i] >= 50)) * (10400 + 25000) * n.lines['length'][i] * 1.2 +
                  ((n.lines['s_nom_opt'][i] <= 88) & (n.lines['s_nom_opt'][i] >= 67)) * (14900 + 25000) * n.lines['length'][i] * 1.2)

    else:
        for i in n.generators.index.tolist():
            s += sum(n.generators_t['p'][i][t] for t in list(n.snapshots)) * n.generators[keyword + "_v"][i]
        for i in n.links.index.tolist():
            s += (n.links['p_nom_opt'][i] - n.links['p_nom'][i]) * n.links[keyword + "_f"][i]
        for i in n.stores.index.tolist():
            s += (n.stores['e_nom_opt'][i] - n.stores['e_nom'][i]) * n.stores[keyword + "_f"][i]
        for i in n.storage_units.index.tolist():
            if n.storage_units_t['p_dispatch'].empty:  # arrivé lors d'une simulation, à enlever si pas besoin
                s += n.storage_units['p_nom'][i] * n.storage_units[keyword + "_f"][i]
            else:
                s += (n.storage_units['p_nom_opt'][i] - n.storage_units['p_nom'][i]) * n.storage_units[keyword + "_f"][i] + sum(
                    n.storage_units_t['p_dispatch'][i][t] for t in list(n.snapshots)) * n.storage_units[keyword + "_v"][i]

    return s

def limit_storage(n, m):
    v_store = sum(m.variables['Store-e_nom'][k] for k in n.stores[n.stores.e_nom_extendable == True].index)
    c_store = sum(n.stores.e_nom[k] for k in n.stores[n.stores.e_nom_extendable == False].index)
    return v_store, c_store
