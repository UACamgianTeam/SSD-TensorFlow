import setuptools

setuptools.setup(
    name="ssd-tensorflow",
    version="0.0.1a",
    author="Ethan Mines",
    packages=["ssd"],
    install_requires=[
        "pycocotools",
        "object-detection",
        "tensorflow",
        "nms",
        "matplotlib",
        "shapely"
    ]
)
