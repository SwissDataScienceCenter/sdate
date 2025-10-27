from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Anomaly detection for Time resolved Computed Tomography'
LONG_DESCRIPTION = ''

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="sdate",
    version=VERSION,
    author="Luis Barba",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    entry_points={
        'console_scripts': [
            'chip-eval-al=chip.evaluation.active_learning:main',
        ]
    }
)