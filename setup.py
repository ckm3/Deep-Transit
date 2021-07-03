# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="deep_transit",
    version="0.0.1",
    author="Kaiming Cui",
    author_email="cuikaiming15@mails.ucas.edu.cn",
    description="Transit detection with object detection algorithm",
    packages=setuptools.find_packages(include=['deep_transit', 'deep_transit.*']),
    # long_description=,
    # long_description_content_type="text/markdown",
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
    install_requires=["lightkurve>=2.0", "torch", "torchvision"],
)