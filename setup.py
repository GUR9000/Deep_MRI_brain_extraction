from setuptools import setup

setup(name='NNet_Core',
      version='1.0',
      description='Tool for brain extraction',
      python_requires='>=3.5',
      license='MIT',
      install_requires=[
      'numpy',
      'nibabel',
      'h5py',
      'theano',
      'pylearn'
      ],
      scripts=['NNet_Core/deep3Dpredict.py','NNet_Core/deep3Dtrain.py'],
      )

