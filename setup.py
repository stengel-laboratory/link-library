import versioneer
import setuptools


setuptools.setup(name='link_library',
      description='Management of crosslink databases',
      author='Kai Kammer',
      author_email='kai-michael.kammer@uni-konstanz.de',
      url='https://git.uni-konstanz.de/kai-michael-kammer/link_library',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=setuptools.find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'numpy',
          'scipy',
          'statsmodels'
      ],
      license='MIT',
      python_requires='>=3.6'
      )
