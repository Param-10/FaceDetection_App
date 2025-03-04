from setuptools import setup, find_packages

setup(
    name="face_detection_webapp",
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
