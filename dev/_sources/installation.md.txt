# Installation

inequality supports Python >= [3.12](https://docs.python.org/3.12/). Please make
sure that you are operating in a supported environment.

## conda

inequality is available through conda:

```shell
conda install -c conda-forge inequality
```

## PyPI

inequality is available on the [Python Package Index](https://pypi.org/project/inequality/).
Therefore, you can either install directly with `pip` from the command line:

```shell
pip install -U inequality
```

or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder. Type:

```shell
pip install .
```

## Installing development version

You may want to use the newest features in the development version of inequality
on GitHub, which may not yet be included in the latest PyPI release. Install the
[pysal/inequality](https://github.com/pysal/inequality) development version from
a command shell:

```shell
pip install git+https://github.com/pysal/inequality.git
```

You can also [fork](https://help.github.com/articles/fork-a-repo/) the
[pysal/inequality](https://github.com/pysal/inequality) repo and create a local
clone of your fork. By making changes to your local clone and submitting a pull
request to [pysal/inequality](https://github.com/pysal/inequality), you can
contribute to inequality development.
