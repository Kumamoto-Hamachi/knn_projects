import os


def make_dir(dir):
    try:
        os.mkdir(dir)
        print(f"make {dir} dir")
    except FileExistsError:
        print(f"{dir} dir have already existed")
