from setuptools import setup, find_packages


requirements = [
    'jupyter',
    'numpy',
    'matplotlib',
    'pandas',
    'torch'
]

setup(
    name='oishani',
    version='0.1.0',
    python_requires='>=3.5',
    author='sushmit roy',
    author_email='sushmit86@gmail.com',
    description='sushmit util',
    packages=find_packages(include=['myutil']),
    zip_safe=True,
    install_requires=requirements,
)