import os
from setuptools import setup, find_packages

# version of the package
__version__ = "v1.0.0"
# fname = os.path.join(
#     os.path.dirname(os.path.realpath(__file__)), "psfhome", "__version__.py"
# )
# print(fname)
# with open(fname, "r") as ff:
#     exec(ff.read())


scripts = []

setup(
    name="psfhome",
    version=__version__,
    description="Impact of PSF Higher Order Moments Error on Weak Lensing",
    author="Tianqing Zhang",
    author_email="tianqinz@andrew.cmu.edu",
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "scipy",
        "galsim",
        "astropy",
        "matplotlib",
    ],
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/LSSTDESC/PSFHOME",
)