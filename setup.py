import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='xlogitprit',
      version='0.0.8',
      description='A Python package for GPU-accelerated estimation of mixed logit models.',
      long_description = long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/PrithviBhatB/xlogit',
      author='Ryan Kelly',
      author_email='ryan@kiiii.com',
      license='MIT',
      packages=['xlogitprit'],
      zip_safe=False,
      python_requires='>=3.5',
      install_requires=[
          'numpy>=1.13.1',
          'scipy>=1.0.0'
      ])
