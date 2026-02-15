import random
import time
import unittest

from pypaq.mpython.mptools import ExProcess, Que, QMessage, sys_res_nfo, MPythonException


class TestMPTools(unittest.TestCase):

    def test_Que(self):
        que = Que()
        self.assertTrue(que.empty)
        self.assertTrue(que.size == 0)

        qm = QMessage(type='test', data=1)
        que.put(qm)
        self.assertTrue(que.size == 1)
        qm = que.get()
        self.assertTrue(qm.data == 1)
        self.assertTrue(que.size == 0)

        self.assertTrue(que.empty)
        self.assertTrue(que.size == 0)
        qm = que.get(block=False)
        self.assertTrue(not qm)

        qm = {'a': 'b'}
        self.assertRaises(MPythonException, que.put, qm)

    def test_ExProcess(self):

        class ExS(ExProcess):

            def exprocess_method(self):
                print(f'---running process---')
                time.sleep(3)

        exs = ExS(oque=Que(), loglevel=10)
        print(f'before spawned: {exs}')
        exs.start()
        print(f'spawned (started): {exs}')
        while exs.alive:
            time.sleep(1)
        print(f'finished: {exs}')
        exs.join()
        print(f'joined: {exs}')
        exs.close()
        print(f'closed: {exs}')

        exs = ExS(oque=Que(), loglevel=10)
        exs.start()
        time.sleep(1)
        exs.terminate()
        print(f'terminated: {exs}')
        exs.close()
        print(f'closed: {exs}')

        exs = ExS(oque=Que(), loglevel=10)
        exs.start()
        time.sleep(1)
        exs.kill()
        print(f'killed: {exs}')
        exs.close()
        print(f'closed: {exs}')

    def test_ExProcess_KeyboardInterrupt(self):

        num_loops = 10

        class ExS(ExProcess):

            def exprocess_method(self):
                cnt = 0
                while cnt < num_loops:
                    print(f'running #{cnt} ..')
                    cnt += 1
                    if random.random() < 0.3:
                        raise KeyboardInterrupt
                    time.sleep(0.3)
                raise KeyboardInterrupt

        exs = ExS(oque=Que(), raise_KeyboardInterrupt=False)
        exs.start()
        msg = exs.oque.get()
        print(msg)
        self.assertTrue(msg.type.startswith('Exception:'))

        exs = ExS(oque=Que(), raise_KeyboardInterrupt=True)
        exs.start()

        # here message won't come since immediate break by keyboard
        #msg = exs.oque.get()
        #print(msg)
        #self.assertTrue(msg is None)

    def test_ExProcess_Exception(self):

        num_loops = 10

        class ExS(ExProcess):

            def exprocess_method(self):
                cnt = 0
                while cnt < num_loops:
                    print(f'running #{cnt} ..')
                    cnt += 1
                    if random.random() < 0.3:
                        raise Exception('RandomException')
                    time.sleep(0.3)
                raise Exception('RandomException')

        exs = ExS(oque=Que(), raise_Exception=False)
        exs.start()
        msg = exs.oque.get()
        print(msg)
        self.assertTrue(msg.type.startswith('Exception:'))

        exs = ExS(oque=Que(), raise_Exception=True)
        exs.start()
        msg = exs.oque.get()
        print(msg)
        self.assertTrue(msg.type.startswith('Exception:'))

    def test_ExProcess_after(self):

        class ExS(ExProcess):

            def exprocess_method(self):
                cnt = 0
                num_loops = 10
                print(f'Starting ExProcess.exprocess_method() loop for {num_loops} loops..')
                while cnt < num_loops:
                    print(f'exprocess_method is running (#{cnt})..')
                    cnt += 1
                    if random.random() < 0.1: raise KeyboardInterrupt
                    if random.random() < 0.1: raise Exception
                    time.sleep(0.3)
                raise Exception

            def after_exception_handle_run(self):
                self.oque.put(QMessage(type='info', data='after'))

        exs = ExS(oque=Que())
        exs.start()
        msg = exs.oque.get()
        print(msg.type)
        self.assertTrue('Exception' in msg.type)
        msg = exs.oque.get()
        print(msg.type, msg.data)
        self.assertTrue(msg.data == 'after')

    def test_sys_res_nfo(self):
        info = sys_res_nfo()
        print(info)
        self.assertTrue(info['cpu_count'] > 0)
        self.assertTrue(info['mem_total_GB'] > 0)
        self.assertTrue(info['mem_used_%'] > 0)