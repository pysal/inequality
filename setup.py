from setuptools import setup

import versioneer

package = "inequality"

description = "inequality: Spatial inequality analysis."

# Fetch README.md for the `long_description`
with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


def _get_requirements_from_files(groups_files):
    """returns a dictionary of all requirements
    keyed by type of requirement.
    Parameters
    ----------
    groups_files : dict
        k - descriptive name, v - file name (including extension)
    Returns
    -------
    groups_reqlist : dict
        k - descriptive name, v - list of required packages
    """

    groups_reqlist = {}
    for k, v in groups_files.items():
        with open(v, "r") as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list
    return groups_reqlist


def setup_package():
    """sets up the python package"""

    # Requirements for: base, dev, docs, plus, and test builds
    _groups_files = {
        "base": "requirements.txt",
        "tests": "requirements_tests.txt",
        "docs": "requirements_docs.txt",
    }
    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop("base")
    extras_reqs = reqs

    setup(
        name=package,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description=description,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pysal/" + package,
        download_url="https://pypi.org/project/" + package,
        maintainer="PySAL Developers",
        maintainer_email="pysal-dev@googlegroups.com",
        keywords="spatial statistics",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        python_requires=">=3.8",
        license="3-Clause BSD",
        packages=[package],
        py_modules=[package],
        install_requires=install_reqs,
        extras_require=extras_reqs,
    )


if __name__ == "__main__":
    setup_package()
