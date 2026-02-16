import pytest
import time

from pypaq.mpython.mptools import ExProcess, Que, QMessage, sys_res_nfo, MPythonException


class ExS(ExProcess):
    def exprocess_method(self):
        print(f'---running process---')
        time.sleep(3)


class ExS_KI(ExProcess):
    def exprocess_method(self):
        cnt = 0
        while cnt < 10:
            print(f'running #{cnt} ..')
            time.sleep(1)
            cnt += 1
            if cnt > 3:
                raise KeyboardInterrupt

class ExS_EX(ExProcess):
    def exprocess_method(self):
        cnt = 0
        while cnt < 10:
            print(f'running #{cnt} ..')
            time.sleep(1)
            cnt += 1
            if cnt > 3:
                raise Exception('SomeException')

class ExS_EXA(ExProcess):

    def exprocess_method(self):
        cnt = 0
        while cnt < 10:
            print(f'running #{cnt} ..')
            time.sleep(1)
            cnt += 1
            if cnt > 3:
                raise Exception('SomeException')

    def after_exception_handle_run(self):
        print("after exception handle called")


def test_Que():
    que = Que()
    assert que.empty
    assert que.size == 0

    qm = QMessage(type='test', data=1)
    que.put(qm)
    assert que.size == 1
    qm = que.get()
    assert qm.data == 1
    assert que.size == 0

    assert que.empty
    assert que.size == 0
    qm = que.get(block=False)
    assert not qm

    with pytest.raises(MPythonException):
        que.put({'a': 'b'})


def test_ExProcess_lifecycle():

    que = Que()
    exs = ExS(oque=que, loglevel=10)
    print(f'before spawned: {exs}')
    exs.start()
    print(f'spawned (started): {exs}')
    exs.join()
    print(f'joined: {exs}')
    exs.close()
    print(f'closed: {exs}')

    assert que.size == 0


def test_ExProcess_kill():

    que = Que()
    exs = ExS(oque=que, loglevel=10)
    exs.start()
    time.sleep(1)
    exs.kill_and_close()
    print(f'killed and closed: {exs}')

    assert que.size == 0


def test_ExProcess_KeyboardInterrupt():

    que = Que()
    exs = ExS_KI(oque=que)
    exs.start()

    msg = que.get()
    print(msg)
    assert msg.type.startswith('Exception:')


def test_ExProcess_Exception():

    que = Que()
    exs = ExS_EX(oque=que)
    exs.start()

    msg = que.get()
    print(msg)
    assert msg.type.startswith('Exception:')


def test_ExProcess_Exception_after():

    que = Que()
    exs = ExS_EXA(oque=que)
    exs.start()

    msg = que.get()
    print(msg)
    assert msg.type.startswith('Exception:')


def test_sys_res_nfo():
    info = sys_res_nfo()
    print(info)
    assert info['cpu_count'] > 0
    assert info['mem_total_GB'] > 0
    assert info['mem_used_%'] > 0
