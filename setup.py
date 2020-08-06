from setuptools import setup, find_packages

install_requires = ['pip',
                    'python>=3.7',
                    'dash>=1.4.1',
                    'flask>=1.1',
                    # 'dash-renderer',
                    # 'dash-html-components',
                    # 'dash-core-components',
                    # 'dash-table',
                    # 'plotly',
                    'pyserial>=3.4',
                    'numpy>=1.9',
                    'pillow>=7.2',
                    ]
__packagename__ = "MultiSpectralCtrl"

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
