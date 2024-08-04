from setuptools import setup, find_packages
import os

def read_requirements(filename):
    """Read dependencies from a requirements file."""
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name='fisheye_3d_reconstruction',
    version='0.1.0',
    author='Jamie Milsom',
    author_email='jamieamilsom@gmail.com',
    description='A tool for calibrating cameras and performing 3D reconstruction with ultra-wide fisheye cameras',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jamiemilsom/fisheye_3d_reconstruction',
    license='MIT',
    packages=find_packages(where='src'), 
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'pytest'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Linux',
    ],
    python_requires='>=3.7',
)
