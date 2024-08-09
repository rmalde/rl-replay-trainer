from setuptools import setup, find_packages
import subprocess
import os

# you'll need to manually install rlgym_sim
# pip install git+https://github.com/AechPro/rocket-league-gym-sim@main

setup(
    name='replay_trainer',  # Replace with your package name
    version='0.1.0',
    description='Train a model from rocket league replays',
    author='rmalde',
    url='https://github.com/rmalde/rl-replay-trainer',  # Replace with your project's URL
    packages=find_packages(),
    install_requires=[
        'torch', 'numpy', 'tqdm', 'rich', 'wandb',
        'python-dotenv', 'scikit-learn'
    ],
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license if different
        'Operating System :: OS Independent',
    ],
)