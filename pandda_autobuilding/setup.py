import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]

setuptools.setup(
    name="autobuild",
    version="0.0.1",
    author="Conor Francis Wild",
    author_email="conor.wild@sky.com",
    description="A package for autobuilding into pandda",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements(),
)