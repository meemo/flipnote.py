from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

setup(name="flipnote",
      version="0.1.0",
      description="A Python library for Flipnote Studio (3D) files",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Meemo",
      author_email="meemo4556@gmail.com",
      license="MIT",
      install_requires=[
          "numpy"
      ],
      url="https://github.com/meemo/flipnote.py",
      project_urls={
          "Bug Tracker": "https://github.com/meemo/flipnote.py/issues",
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      python_requires=">=3.7")
