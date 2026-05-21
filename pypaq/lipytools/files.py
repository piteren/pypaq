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
        self._full_path = Path(full_path)
        self._files: list[str] = []
        self._subfolders :list[Folder] = []

        for entry in sorted(self._full_path.iterdir()):
            if entry.is_file():
                self._files.append(entry.name)
            elif entry.is_dir() and recursive:
                self._subfolders.append(Folder(entry, recursive=recursive))

    @property
    def name(self) -> str:
        return self._full_path.name

    def _get_files_abs(self) -> list[Path]:
        _p = [self._full_path / fn for fn in self._files]
        for sfd in self._subfolders:
            _p.extend(sfd._get_files_abs())
        return _p

    def get_files(self, relative_path: bool = True) -> list[str]:
        _p = self._get_files_abs()
        if relative_path:
            _p = [p.relative_to(self._full_path) for p in _p]
        return [str(p) for p in _p]

    def _build_tree(self, prefix: str = '') -> list[str]:
        entries: list[tuple[Folder | str, bool]] = (
                [(sf, True) for sf in sorted(self._subfolders, key=lambda f: f.name)] +
                [(fn, False) for fn in sorted(self._files)]
        )
        lines = []
        for i, (entry, is_folder) in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = '└─ ' if is_last else '├─ '
            if is_folder:
                lines.append(f'{prefix}{connector}D {entry.name}/ [{len(entry._subfolders)}/{len(entry._files)}]')
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

    def _get_paths_for_processing(
            self,
            path_processed: Path | str,
            add_ext: str | None = None,
            include: str | Iterable[str] | None = None,
            exclude: str | Iterable[str] | None = None,
    ) -> tuple[list[Path], list[Path]]:

        path_processed = Path(path_processed)

        if type(include) is str:
            include = [include]
        if type(exclude) is str:
            exclude = [exclude]

        files = self.get_files(relative_path=True)
        logger.info(f"got {len(files)} files in {self.name} Folder")

        files_sel = []
        for fp in files:
            if exclude and any(e in fp for e in exclude):
                continue
            if include and not any(i in fp for i in include):
                continue
            files_sel.append(fp)
        if len(files_sel) < len(files):
            logger.info(f"selected {len(files_sel)} files for processing using exclude/include")

        files = [self._full_path / fp for fp in files_sel]
        files_target = [path_processed / fp for fp in files_sel]
        if add_ext:
            files_target = [fp.with_name(fp.name + add_ext) for fp in files_target]

        return files, files_target

    def process_files(
            self,
            path_processed: Path | str,
            add_ext: str | None = None,
            include: str | Iterable[str] | None = None,
            exclude: str | Iterable[str] | None = None,
            prog_refresh_delay: int = 60,
            **processing_func_kwargs,
    ):
        """processes all files in the Folder
        (and subfolders if Folder is recursive)
        using processing_func() <- that needs to be implemented

        add_ext - adds additional extension to every processed file
        include - includes only files which got in path any from given patterns
        exclude - excludes files which got in path any from given patterns"""

        logger.info(f"processing files ({len(self._files)}):")
        logger.info(f"> from {self._full_path}")
        logger.info(f"> to {path_processed}")
        logger.info(f"> with {self.processing_func}")
        if add_ext:
            logger.info(f">> adding ext: {add_ext}")
        if include:
            logger.info(f">> including {include}")
        if exclude:
            logger.info(f">> excluding {exclude}")

        files, files_target = self._get_paths_for_processing(
            path_processed=path_processed,
            add_ext=add_ext,
            include=include,
            exclude=exclude,
        )

        for fp in files_target:
            prep_folder(fp)

        prog = ProgBar(len(files), refresh_delay=prog_refresh_delay, logger=logger)
        for fs, ft in zip(files, files_target):
            self.processing_func(
                file_source_path=fs,
                file_target_path=ft,
                **processing_func_kwargs,
            )
            prog.inc(prefix=f"processed {fs.relative_to(self._full_path)}")

    def __str__(self) -> str:
        sl  = f'D {self.name}/ [{len(self._subfolders)}/{len(self._files)}]'
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
        files = [str(fd._full_path / fn) for fn in fd._files]
        for sfd in fd._subfolders:
            files.extend(_get_fd_and_sub_files(sfd))
        return sorted(files)

    return _get_fd_and_sub_files(Folder(fd_path, recursive))


def get_requirements(file_path :str = 'requirements.txt') -> list[str]:
    file_text = r_text(file_path)
    file_lines = file_text.split('\n')
    return [l.strip() for l in file_lines]
