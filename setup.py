from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt", 'r') as fp:
    requirements = fp.readlines()

setup(
    name="pixar_iqa",
    scripts=["run"],
    version="0.1.0",
    description="Pixar Image Quality Analysis",
    long_description=long_description,
    license='APACHE',
    long_description_content_type="text/markdown",
    author="Sandip Dey",
    author_email="sandip.dey1988@yahoo.com",
    packages=['src'],
    include_package_data=True,
    install_requires=requirements,
    platforms=["linux", "unix"],
    python_requires=">3.5.2",
)
