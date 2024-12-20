import setuptools

with open("Readme.md", "r", encoding= "utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Deep Learning Traffic Signs classifier"
AUTHOR_USER_NAME = "Sudhir Joon"
SRC_REPO = "cnnClassifier"

setuptools.setup(
    name= SRC_REPO,
    version= __version__,
    author= AUTHOR_USER_NAME,
    author_email="your_email@example.com",
    description="Deep Learning Traffic Signs classifier using VGG16 Transfer learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
)