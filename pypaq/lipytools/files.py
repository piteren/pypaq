import csv
from dataclasses import dataclass, field
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import yaml

from pypaq.exception import PyPaqException


@dataclass
class Folder:
    name: str # folder name, NOT a full path
    path: str # folder path (folder full_path = path/name)
    subfolders: list["Folder"] = field(default_factory=list) # subfolders names, NOT full paths
    files: list[str] = field(default_factory=list) # files names, NOT full paths

    def _build_tree(self, lines: list[str], prefix: str) -> None:
        entries: list[tuple] = (
            [(sf, True) for sf in sorted(self.subfolders, key=lambda f: f.name)] +
            [(fn, False) for fn in sorted(self.files)]
        )
        for i, (entry, is_folder) in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = '└─ ' if is_last else '├─ '
            if is_folder:
                lines.append(f'{prefix}{connector}D {entry.name}/')
                entry._build_tree(lines, prefix + ('   ' if is_last else '│  '))
            else:
                lines.append(f'{prefix}{connector}f {entry}')

    def __str__(self) -> str:
        lines = [f'D {self.name}/']
        self._build_tree(lines, '')
        return '\n'.join(lines)


def r_text(
        file_path: str | Path,
        raise_exception: bool = False,
) -> str | None:
    if not Path(file_path).is_file():
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def w_text(
        text: str,
        file_path: str | Path,
):
    with open(file_path, 'w') as file:
        return file.write(text)


def r_pickle(
        file_path: str | Path,
        obj_type = None,
        raise_exception: bool=False,
):
    """ reads pickle
    if obj_type is given checks for compatibility with given type """
    if not Path(file_path).is_file():
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None

    compressed = False
    with open(file_path, 'rb') as f:
        if f.read(2) == b'\x1f\x8b': # gzip magic number
            compressed = True

    op_fn = gzip.open if compressed else open
    with op_fn(file_path, 'rb') as file:
        obj = pickle.load(file)

    if obj_type:
        if not type(obj) is obj_type:
            raise PyPaqException(f'ERROR: obj from file is not {str(obj_type)} type')
    return obj


def w_pickle(
        obj,
        file_path: str | Path,
        compressed: bool=False,
):
    op_fn = gzip.open if compressed else open
    with op_fn(file_path, 'wb') as file:
        pickle.dump(obj, file)


def r_json(
        file_path: str | Path,
        raise_exception: bool = False,
) -> dict | list | None:
    if not Path(file_path).is_file():
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def w_json(
        data: dict | list,
        file_path: str | Path,
):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def r_jsonl(
        file_path: str | Path,
        raise_exception: bool = False,
) -> list | None:
    if not Path(file_path).is_file():
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def w_jsonl(
        data: list,
        file_path: str | Path,
):
    with open(file_path, 'w', encoding='utf-8') as file:
        for d in data:
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')


def r_jsonl_gz(
        file_path: str | Path,
        raise_exception: bool = False,
) -> list | None:
    if not Path(file_path).is_file():
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def w_jsonl_gz(
        data: list,
        file_path: str | Path,
):
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        for d in data:
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')


def r_csv(
        file_path: str | Path,
        raise_exception: bool = False,
) -> list | None:
    if not Path(file_path).is_file():
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    csv.field_size_limit(sys.maxsize)
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        return [row for row in reader]


def w_csv(
        data: list[list],
        file_path: str | Path,
):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def r_yaml(
        file_path: str | Path,
        raise_exception: bool = False,
):
    if not Path(file_path).is_file():
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path) as file:
        return yaml.load(file, yaml.Loader)


def extract_folder_path(path: str | Path) -> str:
    """extracts folder full path from a given path"""
    path = Path(path)
    if path.is_dir():
        return str(path)
    return str(path.parent)


def extract_folder_name(path: str | Path) -> str:
    """extracts folder name from a given path"""
    path = Path(path)
    if path.is_dir():
        return path.name
    return path.parent.name


def prep_folder(
        path: str | Path,
        flush_non_empty: bool = False,
):
    """ prepares folder
    path may be a file path -> folder path is extracted """
    folder_path = extract_folder_path(path)
    if folder_path: # in case folder_path == ''
        if flush_non_empty and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)


def list_folder(
        fd_path: str | Path,
        recursive: bool = True,
) -> Folder:
    """lists folder subfolders and files"""
    ls = os.listdir(fd_path)
    fd_name = extract_folder_name(fd_path)
    folder = Folder(name=fd_name, path=fd_path.split(fd_name)[-1])
    for e in ls:
        _full_path = os.path.join(fd_path, e)
        if os.path.isfile(_full_path):
            folder.files.append(e)
        else:
            if recursive:
                folder.subfolders.append(list_folder(_full_path, recursive))
            else:
                folder.subfolders.append(Folder(name=e, path=fd_path))
    return folder


def get_files(
        fd_path: str | Path,
        recursive: bool = True,
) -> list[str]:
    """returns full paths to files form given folder
    recursive: parses also subfolders"""

    def _get_fd_and_sub_files(fd: Folder) -> list[str]:
        files = [f"{fd.path}/{fd.name}/{fn}" for fn in fd.files]
        for sfd in fd.subfolders:
            files.extend(_get_fd_and_sub_files(sfd))
        return files

    folder = list_folder(fd_path, recursive)
    return _get_fd_and_sub_files(folder)


def get_requirements(file_path :str = 'requirements.txt') -> list[str]:
    file_text = r_text(file_path, raise_exception=True)
    file_lines = file_text.split('\n')
    return [l.strip() for l in file_lines]


print(list_folder("pypaq"))