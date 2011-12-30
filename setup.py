# compile with:
# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("cross_cumulant_cy", ["cross_cumulant_cy.pyx"])]

setup(
  name = 'Cross_cumulant_cy',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
