
# setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
 
ext_modules = [
    Extension(
        'bbox',
        sources=['bbox.pyx'],
    ),
	Extension(
        'anchors',
        sources=['anchors.pyx'],
    ),
	Extension(
        'cpu_nms',
        sources=['cpu_nms.pyx'],
    )
]
 
setup(
	name='cython',
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)