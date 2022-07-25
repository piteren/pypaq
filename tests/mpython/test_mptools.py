import random
import time
import unittest

from pypaq.mpython.mptools import sys_res_nfo, ExSubprocess, Que, QMessage


class TestMPTools(unittest.TestCase):

    def test_Que(self):
        que = Que()
        self.assertTrue(que.empty())
        self.assertTrue(que.qsize() == 0)

        qm = QMessage(type='test', data=1)
        que.put(qm)
        qm = que.get()
        self.assertTrue(qm.data == 1)

        self.assertTrue(que.empty())
        self.assertTrue(que.qsize()==0)
        qm = que.get_if()
        self.assertTrue(not qm)

    def test_ExSubprocess_exception(self):

        class ExS(ExSubprocess):

            def subprocess_method(self):
                cnt = 0
                num_loops = 50
                print(f'Starting ExSubprocess.subprocess_method() loop for {num_loops} loops..')
                while True:
                    print(f'subprocess_method is running (#{cnt})..')
                    cnt += 1
                    if random.random() < 0.05: raise KeyboardInterrupt
                    if random.random() < 0.05: raise Exception
                    time.sleep(1)

        exs = ExS(Que(), Que(), raise_unk_exception=False, verb=1)
        exs.start()

    def test_test_ExSubprocess_management(self):

        class ExS(ExSubprocess):

            def subprocess_method(self):
                cnt = 0
                while True:
                    msg = self.ique.get_if()
                    if msg: print(f'ExS received message: {msg}')
                    print(f'subprocess_method is running (#{cnt})..')
                    cnt += 1
                    time.sleep(1)

        class SPManager:

            def __init__(self):
                self.ique = Que()
                self.oque = Que()
                self.exs = ExS(
                    ique=   self.oque,
                    oque=   self.ique,
                    verb=   1)
                self.exs.start()
                self.oque.put(QMessage(type='test', data='data'))
                time.sleep(3)
                print(self.exs.get_info())
                self.exs.kill()
                print(self.exs.get_info())

        SPManager()

    def test_sys_res_nfo(self):
        info = sys_res_nfo()
        print(info)
        self.assertTrue(info['cpu_count'] > 0)
        self.assertTrue(info['mem_total_GB'] > 0)
        self.assertTrue(info['mem_used_%'] > 0)