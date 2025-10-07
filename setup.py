"""
============================================================================
REACH Project - Package Setup Configuration
============================================================================
This file allows the project to be installed as a Python package, making
imports easier across the codebase.

Installation:
  Development mode: pip install -e .
  Regular install: pip install .

Usage after installation:
  from reach.simulation import ArmEnvironment
  from reach.agents import PPOAgent
============================================================================
"""

# from setuptools import setup, find_packages
#
# # Read README for long description
# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()
#
# # Read requirements
# with open("requirements.txt", "r", encoding="utf-8") as fh:
#     requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
#
# setup(
#     name="reach",
#     version="0.1.0",
#     author="REACH Development Team",
#     author_email="tjd352@nau.edu",
#     description="Reinforcement Learning Framework for Assistive Robotic Arm",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/lucaslarson25/reach",
#     packages=find_packages(where="src"),
#     package_dir={"": "src"},
#     classifiers=[
#         "Development Status :: 3 - Alpha",
#         "Intended Audience :: Science/Research",
#         "Topic :: Scientific/Engineering :: Artificial Intelligence",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.9",
#         "Programming Language :: Python :: 3.10",
#     ],
#     python_requires=">=3.9",
#     install_requires=requirements,
#     extras_require={
#         "dev": [
#             "pytest>=7.4.0",
#             "black>=23.0.0",
#             "flake8>=6.0.0",
#         ],
#     },
# )

