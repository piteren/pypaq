import random
import time
import unittest

from pypaq.mpython.mptools import ExProcess, Que, QMessage, sys_res_nfo, MPythonException


class TestMPTools(unittest.TestCase):

    def test_Que(self):
        que = Que()
        self.assertTrue(que.empty)
        self.assertTrue(que.qsize == 0)

        qm = QMessage(type='test', data=1)
        que.put(qm)
        self.assertTrue(que.qsize == 1)
        qm = que.get()
        self.assertTrue(qm.data == 1)
        self.assertTrue(que.qsize == 0)

        self.assertTrue(que.empty)
        self.assertTrue(que.qsize == 0)
        qm = que.get(block=False)
        self.assertTrue(not qm)

        qm = {'a': 'b'}
        self.assertRaises(MPythonException, que.put, qm)

    def test_ExProcess(self):

        class ExS(ExProcess):

            def exprocess_method(self):
                cnt = 0
                num_loops = 5
                while cnt < num_loops:
                    print(f'running #{cnt} ..')
                    cnt += 1
                    time.sleep(0.3)

        exs = ExS(oque=Que(), raise_unk_exception=False, loglevel=10)
        exs.start()

    def test_ExProcess_exception(self):

        class ExS(ExProcess):

            def exprocess_method(self):
                cnt = 0
                num_loops = 10
                while cnt < num_loops:
                    print(f'running #{cnt} ..')
                    cnt += 1
                    if random.random() < 0.05: raise KeyboardInterrupt
                    if random.random() < 0.05: raise Exception('random exception')
                    time.sleep(0.3)
                raise Exception('final exception')

        exs = ExS(oque=Que(), raise_unk_exception=False)
        exs.start()
        msg = exs.oque.get()
        print(msg)

        exs = ExS(oque=Que(), raise_unk_exception=True)
        exs.start()


    def test_ExProcess_after(self):

        class ExS(ExProcess):

            def exprocess_method(self):
                cnt = 0
                num_loops = 10
                print(f'Starting ExProcess.subprocess_method() loop for {num_loops} loops..')
                while cnt < num_loops:
                    print(f'subprocess_method is running (#{cnt})..')
                    cnt += 1
                    if random.random() < 0.1: raise KeyboardInterrupt
                    if random.random() < 0.1: raise Exception
                    time.sleep(0.3)
                raise Exception

            def after_exception_handle_run(self):
                self.oque.put(QMessage(type='info', data='after'))

        exs = ExS(oque=Que(), raise_unk_exception=False)
        exs.start()
        msg = exs.oque.get()
        print(msg.type)
        self.assertTrue('ex_' in msg.type)
        msg = exs.oque.get()
        print(msg.type, msg.data)
        self.assertTrue(msg.data == 'after')

    def test_ExProcess_management(self):

        class ExS(ExProcess):

            def exprocess_method(self):
                cnt = 0
                while True:
                    msg = self.ique.get(block=False)
                    if msg:
                        print(f'ExS received message: {msg}')
                    print(f'subprocess_method is running (#{cnt})..')
                    cnt += 1
                    time.sleep(1)

        exs = ExS(ique=Que(), oque=Que())
        self.assertTrue(not exs.alive)
        self.assertTrue(not exs.closed)

        exs.start()
        exs.ique.put(QMessage(type='test', data='data'))
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