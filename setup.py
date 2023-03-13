"""
Build and install the project.
"""
from setuptools import setup, find_packages

NAME = "NYgrid-python"
FULLNAME = "NYgrid Python Tools"
AUTHOR = "The NYgrid-python Developers"
AUTHOR_EMAIL = "by276@cornell.edu"
MAINTAINER = "Bo Yuan"
MAINTAINER_EMAIL = AUTHOR_EMAIL
LICENSE = "MIT License"
URL = "https://github.com/boyuan276/NYgrid-python"
DESCRIPTION = "A python version of the NYgrid model"
KEYWORDS = "Power system, Renewable energy, Optimization"

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Python :: 3 :: Only",
]
PLATFORMS = "Any"
PACKAGES = find_packages()
SCRIPTS = []
PACKAGE_DATA = {
    # "cmaqpy": ["cmaqpy/data/*"],
}
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "pandas",
    "pyomo",
    "pypower",
    "matplotlib"
]
PYTHON_REQUIRES = ">=3.7"

if __name__ == "__main__":
    setup(
        name=NAME,
        fullname=FULLNAME,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        url=URL,
        platforms=PLATFORMS,
        scripts=SCRIPTS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        python_requires=PYTHON_REQUIRES,
    )
