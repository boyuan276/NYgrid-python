Usage
=====

1. Activate the environment::

       conda activate NYgrid-python

2. Run the model:

   - Go to the ``examples`` folder.

   - Run the base year 2018 case::

         python 01_opf_2018NewParams_daily.py

   - Run the future year 2030 with policy scenarios::

         # 1) 2030BaselineCase
         python 02_opf_2030BaselineCase_daily.py

         # 2) 2030ContractCase
         python 03_opf_2030ContractCase_daily.py

         # 3) 2030StateScenario
         python 04_opf_2030StateScenario_daily.py

   - Note: Generation and load properties and profiles need to be prepared before running these cases.


Helper functions
----------------

1. Functions for running the ``NYGrid`` model, refer to the :ref:`run_nygrid` module.

2. Other utility functions, refer to the :ref:`utils` module.
