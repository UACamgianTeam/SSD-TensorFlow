import setuptools

setuptools.setup(
    name="ssd-tensorflow",
    version="0.0.1a",
    author="Ethan Mines",
    packages=["ssd","common"],
    install_requires=[
        "oriented-object-detection",
        "pycocotools",
        "object-detection",
        "tensorflow",
        "nms",
        "matplotlib",
        "shapely"
    ]
)
