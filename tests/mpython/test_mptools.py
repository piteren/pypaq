import random
import time
import unittest

from pypaq.mpython.mptools import ExSubprocess, Que, QMessage, sys_res_nfo


class TestMPTools(unittest.TestCase):

    def test_Que(self):
        que = Que()
        self.assertTrue(que.empty())
        self.assertTrue(que.qsize() == 0)

        qm = QMessage(type='test', data=1)
        que.put(qm)
        self.assertTrue(que.qsize() == 1)
        qm = que.get()
        self.assertTrue(qm.data == 1)
        self.assertTrue(que.qsize() == 0)

        self.assertTrue(que.empty())
        self.assertTrue(que.qsize() == 0)
        qm = que.get(block=False)
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

        exs = ExS(Que(), Que(), raise_unk_exception=False)
        exs.start()
        msg = exs.oque.get()
        print(msg.type, msg.data)
        self.assertTrue('ex_' in msg.type)


    def test_ExSubprocess_after(self):

        class ExS(ExSubprocess):

            def subprocess_method(self):
                cnt = 0
                print(f'Starting ExSubprocess.subprocess_method() loop..')
                while True:
                    print(f'subprocess_method is running (#{cnt})..')
                    cnt += 1
                    if random.random() < 0.1: raise KeyboardInterrupt
                    if random.random() < 0.1: raise Exception
                    time.sleep(1)

            def after_exception_handle_run(self):
                self.oque.put(QMessage(type='info', data='after'))

        exs = ExS(Que(), Que(), raise_unk_exception=False)
        exs.start()
        msg = exs.oque.get()
        print(msg.type)
        self.assertTrue('ex_' in msg.type)
        msg = exs.oque.get()
        print(msg.type, msg.data)
        self.assertTrue(msg.data == 'after')


    def test_test_ExSubprocess_management(self):

        class ExS(ExSubprocess):

            def subprocess_method(self):
                cnt = 0
                while True:
                    msg = self.ique.get(block=False)
                    if msg: print(f'ExS received message: {msg}')
                    print(f'subprocess_method is running (#{cnt})..')
                    cnt += 1
                    time.sleep(1)

        ique = Que()
        oque = Que()
        exs = ExS(ique=oque, oque=ique)
        self.assertTrue(not exs.alive)
        self.assertTrue(not exs.closed)

        exs.start()
        oque.put(QMessage(type='test', data='data'))
        time.sleep(3)
        print(exs.get_info())
        exs.kill()
        print(exs.get_info())
        self.assertTrue(not exs.alive)
        exs.close()
        self.assertTrue(exs.closed)
        print(exs.get_info())


    def test_sys_res_nfo(self):
        info = sys_res_nfo()
        print(info)
        self.assertTrue(info['cpu_count'] > 0)
        self.assertTrue(info['mem_total_GB'] > 0)
        self.assertTrue(info['mem_used_%'] > 0)