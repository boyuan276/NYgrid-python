.. NYgrid-python documentation master file, created by
   sphinx-quickstart on Thu Dec 12 16:17:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NYgrid-python's documentation!
=========================================

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/boyuan276/NYgrid-python/blob/main/LICENSE
.. image:: https://img.shields.io/badge/python-3.8-blue.svg
.. image:: https://img.shields.io/badge/python-3.9-blue.svg
.. image:: https://github.com/boyuan276/NYgrid-python/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/boyuan276/NYgrid-python/actions/workflows/python-package.yml
.. image:: https://github.com/boyuan276/NYgrid-python/actions/workflows/github-code-scanning/codeql/badge.svg
   :target: https://github.com/boyuan276/NYgrid-python/actions/workflows/github-code-scanning/codeql

A python version of the NYgrid model.

It contains the following components:

- Optimal power flow (OPF) model
- Renewable integration
- Building electrification
- Battery storage
- Electric vehicle (EV) charging

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   data
   usage
   nygrid
   optimizer

.. toctree::
   :hidden:

   allocate
   gen_params
   preprocessing
   run_nygrid
   utils


See also
--------

- `MATLAB version of the NYgrid model <https://github.com/AndersonEnergyLab-Cornell/NYgrid.git>`_
