from setuptools import setup, find_packages

setup(
    name="hdvision", # installed package name
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        "console_scripts": [
            'hdvision=main:main',
        ],
    },
    install_requires=[

    ],
    author="Hyeongdo Lee",
    author_email='mylovercorea@gmail.com',
    description='A package for your data analysis algorithm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/caerang/cv-tools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
