Getting started
===============

Installation
------------

Install Nicomedia and its dependencies from the repository root:

.. code-block:: console

   python -m pip install -e .

Set ``NICOMEDIA_PATH`` to the repository root when using functions that read
repository-local catalogs or write workflow products.

Double-King profile
-------------------

The maintained example compares the double-King point-spread profile with
TDpy's equivalent evaluation:

.. code-block:: console

   python examples/double_king_compatibility.py --typefileplot png

Nicomedia remains useful when reproducing workflows that import its historical
function names. For new shared array utilities, use TDpy; for gravitational
lensing functions, use Chalcedon.