from setuptools import find_packages, setup
import versioneer

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="physiokinematic",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Physio-kinematic Distance Calculator",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.11",
    license="MIT",
    url="https://github.com/tvwenger/physiokinematic",
)
