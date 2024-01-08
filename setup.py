# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="face-ai-kit",
    version="0.1.2a0",
    description="FaceAIKit is a Python library designed for face detection and recognition application.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Tomas Goldmann",
    author_email="igoldmann@fit.vutbr.cz",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    package_data={'face_ai_kit': ['config/base.yaml']},
    include_package_data=True,
    install_requires=["numpy", "opencv-python", "onnxruntime", "gdown", "confuse", "scikit-image"]
)
