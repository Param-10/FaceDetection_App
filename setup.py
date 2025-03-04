from setuptools import setup, find_packages

setup(
    name="FaceDetection_WebApp",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "numpy",
        "opencv-python-headless",
        "torch",
        "torchvision",
        "deepface",
    ],
)
