from setuptools import setup, find_packages

setup(
    name='mca_bart',
    version='0.1.1-alpha',
    package_dir={"": "src"},
    packages=find_packages("src", exclude=['*test*', '*train*', '*documents*']),
    url='',
    license='Apache',
    author='Nguyen Minh Quang',
    author_email='anonymous',
    description='The multi-channel attention package on top of BART',
    install_requires=open('requirements.txt').readlines(),
    python_requires=">=3.7",
)
