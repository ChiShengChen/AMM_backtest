#!/usr/bin/env python3
"""
Setup script for Steer Intent Backtester.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="steer-intent-backtester",
    version="0.1.0",
    author="Quant Team",
    author_email="team@example.com",
    description="CLMM backtesting system for intent-based dynamic rebalancing strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/steer-intent-backtester",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "steerbt=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
