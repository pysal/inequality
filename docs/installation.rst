.. Installation

Installation
============

inequality supports Python>=`3.11`_. Please make sure that you are
operating in a supported environment.

conda
+++++

inequality is available through conda::

  conda install -c conda-forge inequality

pypi
++++

inequality is available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U inequality

or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .

Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of inequality on github - `pysal/inequality`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/inequality`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/inequality.git

You can  also `fork`_ the `pysal/inequality`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/inequality`_, you can
contribute to inequality development.


.. _3.11: https://docs.python.org/3.11/
.. _Python Package Index: https://pypi.org/project/inequality/
.. _pysal/inequality: https://github.com/pysal/inequality
.. _fork: https://help.github.com/articles/fork-a-repo/
