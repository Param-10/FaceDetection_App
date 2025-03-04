from setuptools import setup, find_packages

setup(
    name="face_detection_webapp",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask==2.3.3",
        "Werkzeug==2.3.7",
        "opencv-python-headless==4.8.0.74",
        "numpy==1.24.3",
        "gunicorn==21.2.0",
    ],
    # Optional dependencies
    extras_require={
        "analysis": [
            "deepface==0.0.79",
            "tensorflow==2.13.0",
            "keras==2.13.1",
        ],
        "performance": [
            "torch==2.0.1",
            "torchvision==0.15.2",
        ]
    }
)
