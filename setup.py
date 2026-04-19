import os
import sys
import subprocess
import shutil
import platform

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

LIBUGOMEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "libugomemo")

if platform.system() == "Darwin":
    LIB_NAME = "libugomemo.dylib"
else:
    LIB_NAME = "libugomemo.so"


class BuildWithNative(build_py):
    """Custom build that compiles libugomemo and bundles the shared library."""

    def run(self):
        if os.path.isdir(LIBUGOMEMO_DIR) and os.path.exists(os.path.join(LIBUGOMEMO_DIR, "Makefile")):
            try:
                subprocess.check_call(["make", "shared"], cwd=LIBUGOMEMO_DIR)
                built = os.path.join(LIBUGOMEMO_DIR, "build", "shared", LIB_NAME)
                dest = os.path.join("src", "flipnote", LIB_NAME)
                if os.path.exists(built):
                    shutil.copy2(built, dest)
                    print(f"Bundled {LIB_NAME} into package")
                else:
                    print(f"Warning: {LIB_NAME} not found after build", file=sys.stderr)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: Could not build libugomemo ({e}). "
                      "C acceleration will not be available.", file=sys.stderr)
        else:
            print("Note: libugomemo source not found. "
                  "Install will use pure Python (clone with --recursive for C acceleration).",
                  file=sys.stderr)

        super().run()


setup(
    name="flipnote",
    version="0.2.0",
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
    package_data={"flipnote": ["libugomemo.*"]},
    cmdclass={"build_py": BuildWithNative},
    python_requires=">=3.7",
)
