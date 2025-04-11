import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from package
def get_version():
    """Get version from package without importing it."""
    with open(os.path.join("ap_bwe", "__init__.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                # Remove leading/trailing whitespace and quotes
                return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"  # Default if not found

setuptools.setup(
    name="ap_bwe",
    version=get_version(),
    author="Ye-Xin Lu, Yang Ai, Hui-Peng Du, Zhen-Hua Ling",
    author_email="",  # Add author email if available
    description="Speech bandwidth extension with parallel amplitude and phase prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yxlu-0102/AP-BWE",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'ap_bwe': ['*.py', '**/*.py', 'configs/*.json'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "torch>=1.9.0",
        "torchaudio",
        "numpy",
        "matplotlib",
        "librosa",
        "soundfile",
        "rich",
        "joblib",
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'ap-bwe=ap_bwe.cli:main',
            'ap-bwe-inference-16k=ap_bwe.inference.inference_16k:main',
            'ap-bwe-inference-48k=ap_bwe.inference.inference_48k:main',
            'ap-bwe-train-16k=ap_bwe.train.train_16k:main',
            'ap-bwe-train-48k=ap_bwe.train.train_48k:main',
        ],
    },
)
