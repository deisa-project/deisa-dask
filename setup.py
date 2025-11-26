# =============================================================================
# Copyright (C) 2015-2023 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written  permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import os

from setuptools import setup, find_namespace_packages


def find_version(*file_paths):
    def read(*parts):
        here = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(here, *parts)) as fp:
            return fp.read().strip()

    import re
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    with open('readme.md', 'r') as f:
        return f.read()


version = find_version("src/deisa/dask", "__version__.py")

setup(name='deisa-dask',
      version=version,

      description='Deisa: Dask-Enabled In Situ Analytics',
      long_description=readme(),
      long_description_content_type='text/markdown',
      license='MIT',

      url='https://github.com/deisa-project/deisa-dask',
      project_urls={
          'Bug Reports': 'https://github.com/deisa-project/deisa-dask/issues',
          'Source': 'https://github.com/deisa-project/deisa-dask',
      },

      author='Benoît Martin',
      author_email='bmartin@cea.fr',

      python_requires='>=3.10',

      keywords='deisa in-situ',

      package_dir={'': 'src'},
      packages=find_namespace_packages(where='src', include=['deisa.dask']),

      install_requires=[
          'dask',
          'distributed'
      ],

      extras_require={
          "test": [
              "pytest",
              "numpy",
              "deisa @ git+https://github.com/deisa-project/deisa@main"
          ]
      },
      test_suite='test',

      classifiers=[
          "Programming Language :: Python :: 3.10",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Development Status :: 3 - Alpha"
      ]
      )
