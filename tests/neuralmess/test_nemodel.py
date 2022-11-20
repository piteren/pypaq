import numpy as np
import unittest

from tests.envy import flush_tmp_dir

from pypaq.mpython.mpdecor import proc_wait
from pypaq.neuralmess.get_tf import tf
from pypaq.neuralmess.nemodel import NEModel
from pypaq.neuralmess.layers import lay_dense

NEMODEL_DIR = f'{flush_tmp_dir()}/nemodel'
NEModel.SAVE_TOPDIR = NEMODEL_DIR

DNA = {'seq_len':20, 'emb_num':33, 'seed':111}


def fwd_lin_graph(
        in_drop: float,
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

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_base_init(self):
        NEModel(
            nngraph=    fwd_lin_graph,
            loglevel=   10,
            in_drop=    0.0)
        NEModel(
            nngraph=    fwd_lin_graph,
            opt_func=   None,
            loglevel=   10,
            in_drop=    0.0)

    def test_init_raises(self):
        self.assertRaises(Exception, NEModel)
        kwargs = dict(name='fwd_lin_graph')
        self.assertRaises(Exception, NEModel, **kwargs)
        kwargs = dict(nngraph=fwd_lin_graph)
        self.assertRaises(Exception, NEModel, **kwargs)

    def test_init(self):
        """
        nnm = NEModel(
            name=           'nemodel_test_A',
            nngraph=        fwd_graph,
            opt_func=       None,
            **DNA)
        nnm.save_ckpt()
        self.assertTrue(nnm['opt_func'] is None and nnm['baseLR'] == 0.003)

        nnm = NEModel(
            name=           'nemodel_test_B',
            nngraph=        fwd_graph,
            **DNA)
        self.assertTrue(nnm['opt_func'] is not None and nnm['baseLR']==0.003 and 'loss' in nnm)
        nnm.save_ckpt()
        """
        nnm = NEModel(
            name=       'nemodel_test_C',
            nngraph=    fwd_lin_graph,
            loglevel=   10)
        self.assertTrue(nnm['seed']==111 and nnm['baseLR']==0.003 and 'loss' in nnm)
        self.assertTrue('loss' not in nnm.get_managed_params())
        nnm.save()

        print('\nsaved, now loading..')

        nnm = NEModel(
            name=       'nemodel_test_C',
            loglevel=   10)
        self.assertTrue(nnm['seq_len']==20 and nnm['emb_num']==33)

    def test_call(self):

        nnm = NEModel(
            name=           'nemodel_test_A',
            nngraph=        fwd_lin_graph,
            **DNA)

        inp = np.random.rand(1,784)

        out = nnm.session.run(
            fetches=    nnm['logits'],
            feed_dict=  {nnm['inp_PH']:inp})
        print(out, type(out))
        self.assertTrue(out.shape == (1,10))

    def test_train(self):

        nnm = LinNEModel(
            name=       'nemodel_test_A',
            nngraph=    fwd_lin_graph,
            **DNA)

        data = {
            'train': {'inp_PH': np.random.rand(10000,784), 'lbl_PH': np.random.randint(0,9,10000)},
            'test':  {'inp_PH': np.random.rand(1000,784), 'lbl_PH': np.random.randint(0,9,1000)}}

        nnm.load_data(data=data)

        nnm.train()

    def test_GX(self):

        @proc_wait
        def saveAB():
            nnm = NEModel(
                name=       'nemodel_test_A',
                nngraph=    fwd_lin_graph)
            nnm.save()
            nnm = NEModel(
                name=       'nemodel_test_B',
                nngraph=    fwd_lin_graph)
            nnm.save()

        # INFO: needs to run saveAB() in a subprocess cause TF elements/graphs do conflict
        saveAB()
        NEModel.gx_saved(
            name_parent_main=           'nemodel_test_A',
            name_parent_scnd=           'nemodel_test_B',
            name_child=                 'nemodel_test_C',
            #do_gx_ckpt=                 False
        )


if __name__ == '__main__':
    unittest.main()