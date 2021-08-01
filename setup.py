# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="Deep-Transit",
    version="0.1.0",
    author="Kaiming Cui",
    author_email="cuikaiming15@mails.ucas.edu.cn",
    description="Transit detection with object detection algorithm",
    packages=setuptools.find_packages(where="src"),
    long_description="""
        # Deep-Transit

        ``Deep-Transit`` is an open-source Python package designed for transit detection with a 2D object detection algorithm.

        ## Installation
        ``Deep-Transit`` is easy to install with pip:
        ```
        pip install deep-transit
        ```
        ## Quickstart

        Please visit the [quickstart page](https://deep-transit.readthedocs.io/en/latest/Quickstart.html) for details.
    """,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    include_package_data=True,
    url="https://github.com/ckm3/Deep-Transit",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"],
    python_requires='>=3.6.1',
    install_requires=["torch", "torchvision", "lightkurve>=2.0", "wotan"],
)