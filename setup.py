from setuptools import setup

setup(name='diaml',
      version='1.1.0a1',
      description='Does It All Machine Learning',
      url='http://github.com/chasedehan/diaml',
      author='Chase DeHan',
      author_email='chasedehan@yahoo.com',
      license='MIT',
      packages=['diaml'],
      zip_safe=False, install_requires=['numpy','pandas', 'sklearn'])