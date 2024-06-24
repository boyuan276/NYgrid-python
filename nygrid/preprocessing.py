"""
Preprocessing functions for NY grid data.

Created: 2023-12-26, by Bo Yuan (Cornell University)
Last modified: 2023-12-26, by Bo Yuan (Cornell University)
"""

def agg_demand_county2bus(demand_inc_county, county2bus):
    """
    County-level consumption to bus-level consumption
    """
    demand_inc_county_erie = demand_inc_county['Erie']
    demand_inc_county_westchester = demand_inc_county['Westchester']
    demand_inc_county_rest = demand_inc_county.drop(columns=['Erie', 'Westchester'])

    county2bus_erie = county2bus[county2bus['NAME'] == 'Erie']
    county2bus_westchester = county2bus[county2bus['NAME'] == 'Westchester']
    county2bus_rest = county2bus[(county2bus['NAME'] != 'Erie') & 
                                (county2bus['NAME'] != 'Westchester')]

    demand_inc_bus_erie = demand_inc_county_erie.to_frame()
    demand_inc_bus_erie['55'] = demand_inc_bus_erie['Erie'] * 0.5
    demand_inc_bus_erie['57'] = demand_inc_bus_erie['Erie'] * 0.125
    demand_inc_bus_erie['59'] = demand_inc_bus_erie['Erie'] * 0.375
    demand_inc_bus_erie = demand_inc_bus_erie.drop(columns=['Erie'])
    demand_inc_bus_erie.columns = demand_inc_bus_erie.columns.astype(int)

    demand_inc_bus_westchester = demand_inc_county_westchester.to_frame()
    demand_inc_bus_westchester['74'] = demand_inc_bus_westchester['Westchester'] * 0.5
    demand_inc_bus_westchester['78'] = demand_inc_bus_westchester['Westchester'] * 0.5
    demand_inc_bus_westchester = demand_inc_bus_westchester.drop(columns=['Westchester'])
    demand_inc_bus_westchester.columns = demand_inc_bus_westchester.columns.astype(int)

    county_bus_alloc_rest = county2bus_rest.set_index('NAME').to_dict()['busIdx']
    demand_inc_bus_rest = demand_inc_county_rest.T.groupby(county_bus_alloc_rest).sum().T

    demand_inc_bus = demand_inc_bus_rest.add(demand_inc_bus_erie, fill_value=0)
    demand_inc_bus = demand_inc_bus.add(demand_inc_bus_westchester, fill_value=0)

    return demand_inc_bus
