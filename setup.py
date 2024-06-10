
from setuptools import setup, find_packages

setup(
    name='consentiment',
    version='0.4',
    packages=find_packages(),
    install_requires=[
        'nervaluate==0.1.8'
    ],
    author='Guilherme Marcon',
    author_email='guilherme.santos.marcon@alumni.usp.br',
    description='The library for the ConSentiment graph regularization methods.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GuilhermeMarcon/ConSentiment',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
