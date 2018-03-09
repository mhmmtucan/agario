from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='agar-io-machine-learning',
      author='Muhammet Ucan',
      author_email='mhmmtucan@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'tensorflow'
      ],
      zip_safe=False)