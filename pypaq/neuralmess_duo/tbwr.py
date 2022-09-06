"""

 2022 (c) piteren

    TB (TensorBoard) writer wrapped with separate subprocess

"""

from pypaq.mpython.mptools import ExSubprocess, Que, QMessage

"""
# TensorBoard writer
class TBwr:

    # Internal Process (for TB) with all TF inside (separated from external process)
    class TB_IP(ExSubprocess):

        FLUSH_MSG =  QMessage(type='flush',  data=None)
        POISON_MSG = QMessage(type='poison', data=None)

        def __init__(
                self,
                logdir: str,
                set_to_CPU: bool,
                flush_secs: int):
            ExSubprocess.__init__(self, ique=Que(), oque=Que())
            self.logdir = logdir
            self.set_to_CPU = set_to_CPU
            self.flush_secs = flush_secs

        # method called while starting process (ExSubprocess), processes messages for TB
        def subprocess_method(self):

            import os
            import warnings

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            import tensorflow as tf

            if self.set_to_CPU:
                # check for TF visible_physical_devices and set only CPU as visible
                visible_physical_devices = tf.config.list_physical_devices()
                set_visible = []
                for dev in visible_physical_devices:
                    if 'CPU' in dev.name:
                        set_visible.append(dev)
                tf.config.set_visible_devices(set_visible)

            # INFO: tf.summary.create_file_writer creates logdir while init, because of that self.sw init has moved to the first call of add()
            sw = None
            while True:

                msg = self.ique.get()

                if msg.type == 'poison': break

                if msg.type == 'flush' and sw: sw.flush()

                if msg.type == 'add':
                    if not sw:
                        sw = tf.summary.create_file_writer(
                            logdir=         self.logdir,
                            flush_millis=   1000 * self.flush_secs)
                    with sw.as_default():
                        tf.summary.scalar(
                            name=   msg.data['tag'],
                            data=   msg.data['value'],
                            step=   msg.data['step'])

        def exit(self):
            self.ique.put(TBwr.TB_IP.POISON_MSG)


    def __init__(
            self,
            logdir: str,
            set_to_CPU=     True,
            flush_secs=     10):
        self.tb_ip = TBwr.TB_IP(
            logdir=     logdir,
            set_to_CPU= set_to_CPU,
            flush_secs= flush_secs)
        self.tb_ip.start()

    def add(self,
            value,
            tag: str,
            step: int):
        data = {'tag':tag, 'value':value, 'step':step}
        self.tb_ip.ique.put(QMessage(type='add', data=data))

    def flush(self):
        self.tb_ip.ique.put(TBwr.TB_IP.FLUSH_MSG)

    def exit(self):
        self.tb_ip.ique.put(TBwr.TB_IP.POISON_MSG)
"""

import tensorflow as tf


# TensorBoard writer
class TBwr:

    def __init__(
            self,
            logdir: str,
            set_to_CPU=     True,
            flush_secs=     10):

        # TODO: is this version CPU-GPU-config compatible??
        # TODO: INFO: tf.summary.create_file_writer creates logdir while init, because of that self.sw init has moved to the first call of add()
        self.sw = tf.summary.create_file_writer(
            logdir=         logdir,
            flush_millis=   1000 * flush_secs)

    def add(self,
            value,
            tag: str,
            step: int):
        with self.sw.as_default():
            tf.summary.scalar(
                name=tag,
                data=value,
                step=step)

    def flush(self):
        self.sw.flush()

    def exit(self):
        pass