import os
import re


def create_versioned_dir(path, name):
    if not os.path.isdir(path):
        os.mkdir(path)
    folder = os.path.join(path, name)
    print(folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    folders = list(filter(lambda file: re.search(r'version_[0-9]+', file), os.listdir(folder)))
    print(folders)
    if len(folders) == 0:
        version_dir = os.path.join(folder, 'version_0')
        os.mkdir(version_dir)
        return version_dir
    else:
        versions = [int(f.split('_')[1]) for f in folders]
        versions.sort()
        new_version_dir = os.path.join(folder, f'version_{versions[-1] + 1}')
        os.mkdir(new_version_dir)
        return new_version_dir