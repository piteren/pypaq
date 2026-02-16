import os

import pytest

from tests.envy import flush_tmp_dir

from pypaq.lipytools.files import (
    r_text, w_text,
    r_pickle, w_pickle,
    r_json, w_json,
    r_jsonl, w_jsonl,
    r_jsonl_gz, w_jsonl_gz,
    r_csv, w_csv,
    r_yaml,
    get_dir, prep_folder, list_dir, get_files,
)
from pypaq.exception import PyPaqException

TMP_DIR = f'{flush_tmp_dir()}/files_test'


def setup_function():
    prep_folder(TMP_DIR, flush_non_empty=True)


def test_text():
    fp = f'{TMP_DIR}/test.txt'
    w_text('hello world', fp)
    assert r_text(fp) == 'hello world'


def test_text_missing():
    assert r_text(f'{TMP_DIR}/nonexistent.txt') is None
    with pytest.raises(FileNotFoundError):
        r_text(f'{TMP_DIR}/nonexistent.txt', raise_exception=True)


def test_pickle():
    fp = f'{TMP_DIR}/test.pkl'
    data = {'a': 1, 'b': [2, 3]}
    w_pickle(data, fp)
    loaded = r_pickle(fp)
    assert loaded == data


def test_pickle_compressed():
    fp = f'{TMP_DIR}/test.pkl.gz'
    data = {'a': 1, 'b': [2, 3]}
    w_pickle(data, fp, compressed=True)
    loaded = r_pickle(fp)
    assert loaded == data


def test_pickle_type_check():
    fp = f'{TMP_DIR}/test.pkl'
    w_pickle([1, 2, 3], fp)
    assert r_pickle(fp, obj_type=list) == [1, 2, 3]
    with pytest.raises(PyPaqException):
        r_pickle(fp, obj_type=dict)


def test_pickle_missing():
    assert r_pickle(f'{TMP_DIR}/nonexistent.pkl') is None
    with pytest.raises(FileNotFoundError):
        r_pickle(f'{TMP_DIR}/nonexistent.pkl', raise_exception=True)


def test_json():
    fp = f'{TMP_DIR}/test.json'
    data = {'key': 'value', 'num': 42}
    w_json(data, fp)
    loaded = r_json(fp)
    assert loaded == data


def test_json_list():
    fp = f'{TMP_DIR}/test.json'
    data = [1, 2, 3]
    w_json(data, fp)
    assert r_json(fp) == data


def test_json_missing():
    assert r_json(f'{TMP_DIR}/nonexistent.json') is None
    with pytest.raises(FileNotFoundError):
        r_json(f'{TMP_DIR}/nonexistent.json', raise_exception=True)


def test_jsonl():
    fp = f'{TMP_DIR}/test.jsonl'
    data = [{'a': 1}, {'b': 2}, {'c': 3}]
    w_jsonl(data, fp)
    loaded = r_jsonl(fp)
    assert loaded == data


def test_jsonl_missing():
    assert r_jsonl(f'{TMP_DIR}/nonexistent.jsonl') is None
    with pytest.raises(FileNotFoundError):
        r_jsonl(f'{TMP_DIR}/nonexistent.jsonl', raise_exception=True)


def test_jsonl_gz():
    fp = f'{TMP_DIR}/test.jsonl.gz'
    data = [{'a': 1}, {'b': 2}, {'c': 3}]
    w_jsonl_gz(data, fp)
    loaded = r_jsonl_gz(fp)
    assert loaded == data


def test_jsonl_gz_missing():
    assert r_jsonl_gz(f'{TMP_DIR}/nonexistent.jsonl.gz') is None
    with pytest.raises(FileNotFoundError):
        r_jsonl_gz(f'{TMP_DIR}/nonexistent.jsonl.gz', raise_exception=True)


def test_csv():
    fp = f'{TMP_DIR}/test.csv'
    data = [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]
    w_csv(data, fp)
    loaded = r_csv(fp)
    assert loaded == data


def test_csv_missing():
    assert r_csv(f'{TMP_DIR}/nonexistent.csv') is None
    with pytest.raises(FileNotFoundError):
        r_csv(f'{TMP_DIR}/nonexistent.csv', raise_exception=True)


def test_yaml():
    fp = f'{TMP_DIR}/test.yaml'
    # write yaml manually since there's no w_yaml
    import yaml
    data = {'key': 'value', 'num': 42, 'list': [1, 2, 3]}
    with open(fp, 'w') as f:
        yaml.dump(data, f)
    loaded = r_yaml(fp)
    assert loaded == data


def test_yaml_missing():
    assert r_yaml(f'{TMP_DIR}/nonexistent.yaml') is None
    with pytest.raises(FileNotFoundError):
        r_yaml(f'{TMP_DIR}/nonexistent.yaml', raise_exception=True)


def test_get_dir():
    assert get_dir('/home/user/file.txt') == '/home/user'
    assert get_dir('/home/user/folder') == '/home/user/folder'
    assert get_dir('/home/user/folder/') == '/home/user/folder/'


def test_prep_folder():
    fd = f'{TMP_DIR}/new_folder'
    prep_folder(fd)
    assert os.path.isdir(fd)

    # create a file inside
    w_text('test', f'{fd}/test.txt')
    assert os.path.isfile(f'{fd}/test.txt')

    # flush
    prep_folder(fd, flush_non_empty=True)
    assert os.path.isdir(fd)
    assert not os.path.isfile(f'{fd}/test.txt')


def test_prep_folder_from_file_path():
    fp = f'{TMP_DIR}/sub/deep/file.txt'
    prep_folder(fp)
    assert os.path.isdir(f'{TMP_DIR}/sub/deep')


def test_list_dir():
    fd = f'{TMP_DIR}/list_test'
    prep_folder(fd)
    w_text('a', f'{fd}/a.txt')
    w_text('b', f'{fd}/b.txt')
    prep_folder(f'{fd}/subdir')

    result = list_dir(fd)
    assert sorted(result['files']) == ['a.txt', 'b.txt']
    assert result['dirs'] == ['subdir']


def test_get_files():
    fd = f'{TMP_DIR}/get_files_test'
    prep_folder(fd)
    w_text('a', f'{fd}/a.txt')
    prep_folder(f'{fd}/sub')
    w_text('b', f'{fd}/sub/b.txt')

    files = get_files(fd, recursive=True)
    assert len(files) == 2
    assert any('a.txt' in f for f in files)
    assert any('b.txt' in f for f in files)

    files_flat = get_files(fd, recursive=False)
    assert len(files_flat) == 1
    assert any('a.txt' in f for f in files_flat)
