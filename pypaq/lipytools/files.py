import csv
from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
import pickle
import shutil
import sys
import yaml

from pypaq.exception import PyPaqException


@dataclass
class Folder:
    full_path: Path
    subfolders: list["Folder"] = field(default_factory=list)
    files: list[str] = field(default_factory=list) # file names only, NOT full paths

    @property
    def name(self) -> str:
        return self.full_path.name

    def _build_tree(self, prefix: str = '') -> list[str]:
        entries: list[tuple[Folder | str, bool]] = (
            [(sf, True) for sf in sorted(self.subfolders, key=lambda f: f.name)] +
            [(fn, False) for fn in sorted(self.files)]
        )
        lines = []
        for i, (entry, is_folder) in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = '└─ ' if is_last else '├─ '
            if is_folder:
                lines.append(f'{prefix}{connector}D {entry.name}/')
                lines.extend(entry._build_tree(prefix + ('   ' if is_last else '│  ')))
            else:
                lines.append(f'{prefix}{connector}f {entry}')
        return lines

    def __str__(self) -> str:
        return '\n'.join([f'D {self.name}/'] + self._build_tree())

    @classmethod
    def from_path(cls, fd_path: str | Path, recursive: bool = True) -> "Folder":
        """builds a Folder tree from the given directory path"""
        fd_path = Path(fd_path)
        folder = cls(full_path=fd_path)
        for entry in fd_path.iterdir():
            if entry.is_file():
                folder.files.append(entry.name)
            elif entry.is_dir():
                if recursive:
                    folder.subfolders.append(cls.from_path(entry, recursive))
                else:
                    folder.subfolders.append(cls(full_path=entry))
        return folder


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
    with open(file_path, 'w', encoding='utf-8') as file:
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

    if obj_type is not None:
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
        return list(reader)


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
    if not path.suffix and not path.is_file():
        return str(path)
    return str(path.parent)


def extract_folder_name(path: str | Path) -> str:
    """extracts folder name from a given path"""
    path = Path(path)
    if not path.suffix and not path.is_file():
        return path.name
    return path.parent.name


def prep_folder(
        path: str | Path,
        flush_non_empty: bool = False,
):
    """ prepares folder
    path may be a file path -> folder path is extracted """
    folder_path = Path(extract_folder_path(path))
    if flush_non_empty and folder_path.is_dir():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def get_files(
        fd_path: str | Path,
        recursive: bool = True,
) -> list[str]:
    """returns full paths to files form given folder
    recursive: parses also subfolders"""

    def _get_fd_and_sub_files(fd: Folder) -> list[str]:
        files = [str(fd.full_path / fn) for fn in fd.files]
        for sfd in fd.subfolders:
            files.extend(_get_fd_and_sub_files(sfd))
        return files

    return _get_fd_and_sub_files(Folder.from_path(fd_path, recursive))


def get_requirements(file_path :str = 'requirements.txt') -> list[str]:
    file_text = r_text(file_path, raise_exception=True)
    file_lines = file_text.split('\n')
    return [l.strip() for l in file_lines]


print(Folder.from_path("pypaq"))
