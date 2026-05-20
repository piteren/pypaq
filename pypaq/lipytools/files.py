import csv
import gzip
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
from typing import Iterable
import yaml

from pypaq.exception import PyPaqException
from pypaq.lipytools.printout import ProgBar

logger = logging.getLogger(__name__)


class Folder:

    def __init__(
            self,
            full_path: Path | str,
            recursive: bool = True,
    ):
        self.full_path = Path(full_path)
        self.files: list[str] = []
        self.subfolders :list[Folder] = []

        for entry in sorted(self.full_path.iterdir()):
            if entry.is_file():
                self.files.append(entry.name)
            elif entry.is_dir() and recursive:
                self.subfolders.append(Folder(entry, recursive=recursive))

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
                lines.append(f'{prefix}{connector}D {entry.name}/ [{len(entry.subfolders)}/{len(entry.files)}]')
                lines.extend(entry._build_tree(prefix + ('   ' if is_last else '│  ')))
            else:
                lines.append(f'{prefix}{connector}f {entry}')
        return lines

    def processing_func(
            self,
            file_source_path: Path | str,
            file_target_path: Path | str,
            **kwargs,
    ):
        raise NotImplementedError("not implemented Folder.processing_func()!")

    def process_files(
            self,
            path_processed: Path | str,
            add_ext: str | None = None,
            exclude: Iterable[str] | None = None,
    ):
        """processes all files in the Folder
        with subfolders if Folder is recursive
        using processing_func() <- that needs to be implemented

        add_ext - adds additional extension to every processed file
        exclude - excludes files which got in path any from given patterns"""

        path_processed = Path(path_processed)
        prep_folder(path_processed)

        logger.info(f"processing {len(self.files)} files")
        logger.info(f"> from {self.name}")
        logger.info(f"> to {path_processed}")
        if add_ext:
            logger.info(f"> adding ext: {add_ext}")
        logger.info(f"> with {self.processing_func}")
        logger.info(f"> excluding {exclude}")

        skipped = 0
        if len(self.files):

            prog = ProgBar(len(self.files), logger=logger)
            for fn in self.files:

                if exclude and any(e in fn for e in exclude):
                    skipped += 1
                    prog.inc(prefix=f"skipped {fn} from {self.name}")
                    continue

                file_target_path = path_processed / fn
                if add_ext:
                    file_target_path = file_target_path.with_name(file_target_path.name + add_ext)

                self.processing_func(
                    file_source_path=self.full_path / fn,
                    file_target_path=file_target_path)
                prog.inc(prefix=f"processed {fn} from {self.name}")

        if skipped:
            logger.info(f">> skipped {skipped} files")

        for sfd in self.subfolders:
            sfd.processing_func = self.processing_func
            sfd.process_files(path_processed / sfd.name, exclude=exclude)

    def __str__(self) -> str:
        sl  = f'D {self.name}/ [{len(self.subfolders)}/{len(self.files)}]'
        return '\n'.join([sl] + self._build_tree())


def r_text(file_path: str | Path) -> str:
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
):
    """ reads pickle
    if obj_type is given checks for compatibility with given type """
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
        pickle.dump(obj, file)  #type: ignore


def r_json(file_path: str | Path) -> dict | list:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def w_json(
        data: dict | list,
        file_path: str | Path,
):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def r_jsonl(file_path: str | Path) -> list:
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


def r_jsonl_gz(file_path: str | Path) -> list:
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


def r_csv(file_path: str | Path) -> list:
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


def r_yaml(file_path: str | Path):
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
    """returns full paths (sorted) to files form a given folder
    recursive: parses also subfolders"""

    def _get_fd_and_sub_files(fd: Folder) -> list[str]:
        files = [str(fd.full_path / fn) for fn in fd.files]
        for sfd in fd.subfolders:
            files.extend(_get_fd_and_sub_files(sfd))
        return sorted(files)

    return _get_fd_and_sub_files(Folder.from_path(fd_path, recursive))


def get_requirements(file_path :str = 'requirements.txt') -> list[str]:
    file_text = r_text(file_path)
    file_lines = file_text.split('\n')
    return [l.strip() for l in file_lines]
