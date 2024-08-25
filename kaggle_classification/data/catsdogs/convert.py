"""Convert to common format."""

from glob import glob
from os import makedirs, remove, rename
from shutil import rmtree, unpack_archive

import cv2
from tqdm import tqdm


def setup() -> None:
    """Create/Clean up directories."""
    rmtree("cat", ignore_errors=True)
    rmtree("dog", ignore_errors=True)
    makedirs("cat")
    makedirs("dog")


def extract() -> None:
    """Extract zip."""
    unpack_archive("microsoft-catsvsdogs-dataset.zip")


def move() -> None:
    """Move folders."""
    rename("PetImages/Cat", "cat")
    rename("PetImages/Dog", "dog")


def cleanup() -> None:
    """Remove corrupt images."""
    for image_path in tqdm(glob("**/*.jpg")):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error:
            remove(image_path)


def teardown() -> None:
    """Delete temporary files."""
    remove("microsoft-catsvsdogs-dataset.zip")
    remove("MSR-LA - 3467.docx")
    remove("readme[1].txt")
    remove("cat/Thumbs.db")
    remove("dog/Thumbs.db")

    rmtree("PetImages", ignore_errors=True)


def run() -> None:
    """Convert to common format."""
    setup()
    extract()
    move()
    cleanup()
    teardown()


if __name__ == "__main__":
    run()
