from setuptools import setup, find_packages

setup(name='discrete-skip-gram',
      version='0.0.1',
      install_requires=['Keras','theano'],
      author='Ben Striner',
      url='https://github.com/bstriner/discrete-skip-gram',
      packages=find_packages())
