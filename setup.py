from setuptools import setup, find_packages

requirements = [
    "numpy",
    "tensorflow",
    "opencv-python",
    "matplotlib",
    "Pillow",
    "scikit-learn",
    "click",
    "tensorflow_datasets"
]
setup(
    name="deepinsight-iqa",
    entry_points={
        'console_scripts': [
            "deepiqa_train=deepinsight_iqa.cli:train",
            "deepiqa_predict=deepinsight_iqa.cli:predict",
            "deepiqa_eval=deepinsight_iqa.cli:evaluate",
            "deepiqa_prepare_tfrecord=deepinsight_iqa.cli:prepare_tf_record"
        ],
    },
    version="0.1.0",
    description="Deep Learning based Image Quality Analysis",
    long_description=open("README.md").read(),
    license='APACHE',
    long_description_content_type="text/markdown",
    author="Sandip Dey",
    author_email="sandip.dey1988@yahoo.com",
    include_package_data=True,
    packages=find_packages(include=['deepinsight_iqa*']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7.5",
)
