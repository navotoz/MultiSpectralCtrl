from setuptools import setup, find_packages

import os
import sys
import platform

# Opencv headless not available on ARM platforms, need to manual install
# if platform.machine() in ["arm", "aarch64", "aarch64_be", "armv8b", "armv8l"]:
#     install_requires=['pyserial', 'tqdm', 'numpy', 'natsort', 'pyudev', 'psutil', 'pyusb', 'pyftdi']
#
#     print("System detected as ARM. This library depends on OpenCV, which is not \
#            available as a wheel yet so you will need to build from scratch. If you're running \
#            aarch64, you can try 'pip install opencv-python-aarch64' but this is not officially supported.")
# else:
#     install_requires=['pyserial', 'opencv-python-headless', 'tqdm', 'numpy',
#                       'pyudev', 'psutil', 'natsort', 'pyusb', 'pyftdi']

install_requires = ['pip',
                    'python>=3.7',
                    'dash>=1.4.1',
                    # 'dash-renderer',
                    # 'dash-html-components',
                    # 'dash-core-components',
                    # 'dash-table',
                    # 'plotly',
                    'pyserial>=3.4',
                    'numpy>=1.9',
                    'pillow>=7.2',
                    ]
__packagename__ = "hyper_spectral"

setup(
    name=__packagename__,
    version='1.0',
    packages=find_packages(),
    author='Navot Oz',
    author_email='navot@volcani.agri.gov.il',
    license='MIT',
    short_description='Install Vimba Separately.',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # zip_safe=False,
    # include_package_data=True,
    # scripts=['scripts/split_seqs'],
    install_requires=install_requires
)
