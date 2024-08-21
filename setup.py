from setuptools import setup, find_namespace_packages

with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [str(requirement).strip() for requirement in fp]

# scripts = [str(script) for script in Path('scripts').iterdir() if script.is_file()]

setup(
    name="shtoolkit",
    version="1.0.0",
    author="wjh",
    packages=find_namespace_packages(include=["shtoolkit"]),
    install_requires=install_requires,
    python_requires=">=3.11",
    # scripts=scripts
)
