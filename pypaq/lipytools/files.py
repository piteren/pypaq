import csv
import json
import os
import pickle
import shutil
import yaml
from pathlib import Path
import sys
from typing import Union, Dict, List, Optional

from pypaq.exception import PyPaqException


def r_text(file_path:str, raise_exception=False) -> Optional[str]:
    if not os.path.isfile(file_path):
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def w_text(text:str, file_path:str):
    with open(file_path, 'w') as file:
        return file.write(text)


def r_pickle(file_path, obj_type=None, raise_exception=False):
    """ reads pickle
    if obj_type is given checks for compatibility with given type """
    if not os.path.isfile(file_path):
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None

    with open(file_path, 'rb') as file:
        obj = pickle.load(file)

    if obj_type:
        if not type(obj) is obj_type:
            raise PyPaqException(f'ERROR: obj from file is not {str(obj_type)} type')
    return obj


def w_pickle(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def r_json(file_path, raise_exception=False) -> Optional[Union[Dict,List]]:
    if not os.path.isfile(file_path):
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def w_json(data:Union[Dict,List], file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def r_jsonl(file_path, raise_exception=False) -> Optional[List]:
    if not os.path.isfile(file_path):
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def w_jsonl(data:List, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for d in data:
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')


def r_csv(file_path, raise_exception=False) -> Optional[List]:
    if not os.path.isfile(file_path):
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    csv.field_size_limit(sys.maxsize)
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        return [row for row in reader]


def r_yaml(file_path, raise_exception=False):
    if not os.path.isfile(file_path):
        if raise_exception:
            raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path) as file:
        return yaml.load(file, yaml.Loader)


def get_dir(path: Union[str, Path]):
    path = str(path)
    path_split = path.split('/')
    if path_split[-1].find('.') != -1: path_split = path_split[:-1]
    return '/'.join(path_split)


def prep_folder(path:Union[str,Path], flush_non_empty=False):
    """ prepares folder
    path may be a file path -> folder path is extracted """
    folder_path = get_dir(path)
    if folder_path: # in case folder_path == ''
        if flush_non_empty and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)


def list_dir(path: Union[str, Path]) -> Dict[str,List]:
    ls = os.listdir(path)
    lsD = {'files':[], 'dirs':[]}
    for e in ls:
        lsD['files' if os.path.isfile(f'{path}/{e}') else 'dirs'].append(e)
    return lsD


def get_requirements(file_path:str='requirements.txt') -> List[str]:
    file_text = r_text(file_path, raise_exception=True)
    file_lines = file_text.split('\n')
    return [l.strip() for l in file_lines]

