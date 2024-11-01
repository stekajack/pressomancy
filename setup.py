from setuptools import find_packages, setup

setup(
    name="pressomancy",
    version="1.0.0",
    author="Deniz Mostarac",
    author_email="deniz.mostarac@uniroma1.it",
    description="Simulation package wrapping Espresso objects",
    packages=find_packages(),
    test_suite="test",
    include_package_data=True,  # Include non-Python files like those in the resources folder
    install_requires=[
        "numpy",  # Add any pip-installable dependencies here
        # Don't include espresso since it's run via pypresso
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change this if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the Python version requirement
)
