import os
import numpy as np
from distutils import sysconfig
from distutils.core import setup, Extension

# clean previous build
cppfiles = []
for root, dirs, files in os.walk("../", topdown=False):
    for name in files:
        if name == "main.cpp":
            continue
        if os.path.splitext(name)[-1] == ".cpp":
            cppfiles.append(root + '/' + name)

print(cppfiles)

aam_module = Extension(
    'aam',
    libraries=['opencv_world330'],
    sources=cppfiles,
    language="c++",
    library_dirs=[
        "../../opencv/build/x64/vc14/lib"
    ])

setup(
    name='dde_wrapper',
    version='1.0',
    data_files=[
        (sysconfig.get_python_lib(),
         ["../../opencv/build/x64/vc14/bin/opencv_world330.dll",
          "../../opencv/build/x64/vc14/bin/opencv_ffmpeg330_64.dll"])
    ],
    ext_modules=[aam_module],
    include_dirs=[
        np.get_include(),
        "../",
        "../../eigen3.3/",
        "../../opencv/build/include/"])
