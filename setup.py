#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from git2net import __version__

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

setuptools.setup(
    name="git2net",
    version=__version__,
    author="Christoph Gote",
    author_email="cgote@ethz.ch",
    license='AGPL-3.0+',
    description="An OpenSource Python package for the extraction of fine-grained and " +
                "time-stamped co-editing networks from git repositories.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gotec/git2net',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent"
    ],
    test_suite='tests',
    keywords='co-editing networks repository mining network analysis',
    install_requires=install_requirements,
    entry_points={
        "console_scripts": [
            "git2net = git2net.command_line:main",
        ]
    },
    package_data={'git2net': ['helpers/binary-extensions/binary-extensions.json']},
)
