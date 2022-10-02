import numpy as np
import unittest

from tests.envy import flush_tmp_dir

from pypaq.mpython.mpdecor import proc_wait
from pypaq.neuralmess.get_tf import tf
from pypaq.neuralmess.nemodel import NEModel, fwd_graph
from pypaq.neuralmess.layers import lay_dense

NEMODEL_DIR = f'{flush_tmp_dir()}/nemodel'

DNA = {'seq_len':20, 'emb_num':33, 'seed':111}


def fwd_lin_graph(
        in_drop=    0.0,
        in_shape=   784,
        out_shape=  10,
        seed=       121):

    inp_PH = tf.placeholder(
            name=           'inp_PH',
            dtype=          tf.float32,
            shape=          [None, in_shape])

    lbl_PH = tf.placeholder(
            name=           'lbl_PH',
            dtype=          tf.int32,
            shape=          [None])

    inp = inp_PH
    if in_drop:
        inp = tf.layers.dropout(
            inputs=     inp,
            rate=       in_drop,
            seed=       seed)

    logits = lay_dense(
        input=  inp,
        units=  out_shape,
        seed=   seed)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels= lbl_PH,
        logits= logits)
    loss = tf.reduce_mean(loss)

    pred = tf.argmax(logits, axis=-1)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred, dtype=tf.int32), lbl_PH), dtype=tf.float32))

    return {
        'inp_PH':   inp_PH,
        'lbl_PH':   lbl_PH,
        'logits':   logits,
        'loss':     loss,
        'acc':      acc}


class LinNEModel(NEModel):

    def build_feed(self, batch: dict, train=True) -> dict:
        feed = {self['inp_PH']: batch['inp_PH']}
        if train: feed[self['lbl_PH']] = batch['lbl_PH']
        return feed


class TestNEModel(unittest.TestCase):

    def test_init(self):

        flush_tmp_dir()

        nnm = NEModel(
            name=           'nemodel_test_A',
            fwd_func=       fwd_graph,
            opt_func=       None,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,  # INFO: unittests crashes with logger
            verb=           1,
            **DNA)
        nnm.save_ckpt()
        self.assertTrue(nnm['opt_func'] is None and nnm['baseLR'] == 0.003)

        nnm = NEModel(
            name=           'nemodel_test_B',
            fwd_func=       fwd_graph,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,  # INFO: unittests crashes with logger
            verb=           1,
            **DNA)
        self.assertTrue(nnm['opt_func'] is not None and nnm['baseLR']==0.003 and 'loss' in nnm)
        nnm.save_ckpt()

        nnm = NEModel(
            name=           'nemodel_test_C',
            fwd_func=       fwd_graph,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           1,
            **DNA)
        self.assertTrue(nnm['seed']==111 and nnm['baseLR']==0.003 and 'loss' in nnm)
        self.assertTrue('loss' not in nnm.get_managed_params())
        nnm.save()

        print('\nsaved, now loading...')

        nnm = NEModel(
            name=           'nemodel_test_C',
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           0)
        self.assertTrue(nnm['seq_len']==20 and nnm['emb_num']==33)


    def test_call(self):

        flush_tmp_dir()

        nnm = NEModel(
            name=           'nemodel_test_A',
            fwd_func=       fwd_lin_graph,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,  # INFO: unittests crashes with logger
            verb=           1,
            **DNA)

        inp = np.random.rand(1,784)

        out = nnm.session.run(
            fetches=    nnm['logits'],
            feed_dict=  {nnm['inp_PH']:inp})
        print(out, type(out))
        self.assertTrue(out.shape == (1,10))


    def test_train(self):

        flush_tmp_dir()

        nnm = LinNEModel(
            name=           'nemodel_test_A',
            fwd_func=       fwd_lin_graph,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,  # INFO: unittests crashes with logger
            verb=           1,
            **DNA)

        data = {
            'train': {'inp_PH': np.random.rand(10000,784), 'lbl_PH': np.random.randint(0,9,10000)},
            'test':  {'inp_PH': np.random.rand(1000,784), 'lbl_PH': np.random.randint(0,9,1000)}}

        nnm.load_data(data=data)

        nnm.train()


    def test_GX(self):

        flush_tmp_dir()

        @proc_wait
        def saveAB():
            nnm = NEModel(
                name=           'nemodel_test_A',
                fwd_func=       fwd_graph,
                save_topdir=    NEMODEL_DIR,
                do_logfile=     False,      # INFO: unittests crashes with logger
                verb=           1,
                **DNA)
            nnm.save()
            nnm = NEModel(
                name=           'nemodel_test_B',
                fwd_func=       fwd_graph,
                save_topdir=    NEMODEL_DIR,
                do_logfile=     False,      # INFO: unittests crashes with logger
                verb=           1,
                **DNA)
            nnm.save()

        # INFO: needs to run saveAB() in a subprocess cause TF elements/graphs do conflict
        saveAB()
        NEModel.gx_saved(
            name_parent_main=           'nemodel_test_A',
            name_parent_scnd=           'nemodel_test_B',
            name_child=                 'nemodel_test_C',
            save_topdir_parent_main=    NEMODEL_DIR,
            #do_gx_ckpt=                 False
        )


if __name__ == '__main__':
    unittest.main()