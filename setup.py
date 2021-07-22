from setuptools import setup, find_packages


setup(
    name="deepinsight-iqa",
    scripts=["run"],
    version="0.1.0",
    description="Deep Learning based Image Quality Analysis",
    long_description=open("README.md").read(),
    license='APACHE',
    long_description_content_type="text/markdown",
    author="Sandip Dey",
    author_email="sandip.dey1988@yahoo.com",
    packages=['src'],
    include_package_data=True,
    install_requires=open("requirements.txt", 'r').readlines(),
    platforms=["linux", "unix"],
    python_requires=">3.5.2",
)
