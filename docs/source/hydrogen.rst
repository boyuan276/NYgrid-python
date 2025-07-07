Hydrogen integration
====================

This section provides a high level outline for incorporating hydrogen
production and consumption into the NYgrid model. The intention is to
use excess renewable electricity to produce hydrogen via electrolysis,
store or transport the hydrogen, and then dispatch it as a fuel during
periods of network congestion.

Steps for model extension
-------------------------

#. **Add hydrogen storage assets**: Extend the existing ``esr_prop.csv``
   file in :mod:`data/grid` with additional entries representing the
   electrolyzer and hydrogen storage systems. The columns defined in the
   CSV file map directly to the attributes used by
   :func:`nygrid.nygrid.convert_esr_2_gen` for energy storage resources.
   Example columns include ``ESR_CRG_MAX`` and ``ESR_DIS_MAX`` which
   correspond to charging and discharging limits.

#. **Define generator parameters**: If hydrogen is used in dedicated
   generators (e.g. fuel cells), add corresponding rows to
   ``gen_prop.csv`` and ``genfuel_prop.csv``. These files describe the
   static generator characteristics such as ``PMAX`` and fuel type, as
   loaded by :func:`nygrid.run_nygrid.read_grid_prop`.

#. **Update operational profiles**: Provide time series profiles for the
   electrolyzer load and hydrogen generator availability similar to the
   ``write_gen_prop_profiles_`` notebooks in the ``examples`` directory.
   These profiles are then consumed by ``set_gen_max_sch`` and
   ``set_gen_min_sch`` within :class:`nygrid.nygrid.NYGrid`.

#. **Adjust model costs and efficiencies**: Use ``ESR_CRG_EFF`` and
   ``ESR_DIS_EFF`` to represent electrolysis and fuel cell efficiencies.
   Charging and discharging cost fields allow the model to value
   hydrogen production against other resources.

With these additions the hydrogen facilities will be treated similarly
to existing storage resources (battery and pumped storage) and can be
scheduled by the optimizer to absorb excess renewable energy and produce
power during constrained hours.
