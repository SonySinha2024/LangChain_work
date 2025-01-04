## set up
from setuptools import setup, find_packages

setup(
    name='mlapps',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'uvicorn==0.22.0',
        'python-dotenv==1.0.0',
        'google-generativeai>=0.7.0,<0.8.0',
        'pydantic>=2.8.2',
        'requests==2.32.3',
        'fastapi==0.114.0',
        'python-multipart==0.0.9',
        'psycopg2-binary== 2.9.9',
        'pydantic-extra-types>=2.9.0',
        'pydantic-settings>=2.2.1',
        'psycopg2 ==2.9.9',
        'RapidFuzz==3.10.0',
        'requests==2.32.3',
        'regex==2024.9.11'
    ],
    author='ML',
    description='MLApps',
    url='https://thespidercircle@bitbucket.org/radixile/constructionfirst/machine-learning/fin-bot.git',
    classifiers=[
        'Programming Language ::  Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
