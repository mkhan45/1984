import shutil
from pathlib import Path

folder = Path(input("Path of the parent directory:\n> "))

for dir in folder.iterdir():
    if dir.is_dir():
        for image in dir.iterdir():
            shutil.move(str(image), str(folder))
        dir.rmdir()
