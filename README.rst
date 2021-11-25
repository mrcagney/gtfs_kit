GTFS Kit
********
.. image:: https://travis-ci.com/mrcagney/gtfs_kit.svg?branch=master
    :target: https://travis-ci.come/mrcagney/gtfs_kit

GTFS Kit is a Python 3.7+ kit for analyzing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It uses Pandas and Shapely to do the heavy lifting.

This project supersedes `GTFSTK <https://github.com/mrcagney/gtfstk>`_.


Installation
=============
``pip install gtfs_kit``.


Examples
========
Examples are in the Jupyter notebook ``notebooks/examples.ipynb``.


Authors
=========
- Alex Raichev, 2019-09


Documentation
=============
On Github Pages `here <https://mrcagney.github.io/gtfs_kit_docs>`_.


Notes
=====
- Development status is Alpha
- This project uses semantic versioning
- Thanks to `MRCagney <http://www.mrcagney.com/>`_ for donating to this project
- Constructive feedback and code contributions welcome. Please issue pull requests into the ``develop`` branch and include tests.
- GTFS time is measured relative noon minus 12 hours, which can mess things up when crossing into daylight savings time. I don't think this issue causes any bugs in GTFS Kit, but you and i have been warned. Thanks to derhuerst for bringing this to my attention in `closed Issue 8 <https://github.com/mrcagney/gtfs_kit/issues/8#issue-1063633457>`_.
