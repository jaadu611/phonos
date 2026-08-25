from setuptools import setup, find_packages

setup(
    name="audioforge",
    version="0.1.0",
    description="Forensic audio separation and analysis toolkit",
    author="your-username",
    packages=find_packages(),
    py_modules=["cli"],
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "audioforge=cli:main",
        ],
    },
)
