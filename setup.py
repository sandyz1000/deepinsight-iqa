from setuptools import setup

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
    packages=['deepinsight_iqa'],
    include_package_data=True,
    install_requires=requirements,
    platforms=["linux", "unix"],
    python_requires=">3.7.5",
)
