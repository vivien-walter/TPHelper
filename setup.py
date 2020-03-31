import os
from setuptools import setup

# Read the README file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

try:
    import microImage
except:
    raise Exception("Installing this module requires the module microImage to be installed. Please download it from https://github.com/vivien-walter/microImage and install it.")

setup(
    name = "TPHelper",
    version = "1.0",
    author = "Vivien WALTER",
    author_email = "walter.vivien@gmail.com",
    description = ("Python3 module to define and optimize parameters for TrackPy."),
    license = "BSD",
    keywords = "tracking object image open",
    url = "https://github.com/vivien-walter/TPHelper",
    packages=['TPHelper'],
    long_description=read('README.md'),
    classifiers=[
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.2',
    install_requires=[
        'trackpy',
    ]
)
