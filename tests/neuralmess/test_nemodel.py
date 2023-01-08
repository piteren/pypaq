import numpy as np
import unittest

from tests.envy import flush_tmp_dir

from pypaq.mpython.mpdecor import proc_wait
from pypaq.neuralmess.get_tf import tf
from pypaq.neuralmess.nemodel import NEModel
from pypaq.neuralmess.layers import lay_dense

NEMODEL_DIR = f'{flush_tmp_dir()}/nemodel'
NEModel.SAVE_TOPDIR = NEMODEL_DIR


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

    def test_save_load(self):
        model = NEModel(
            nngraph=    fwd_lin_graph,
            loglevel=   10,
            in_drop=    0.1)
        self.assertTrue(model['seed']==121 and model['baseLR']==0.0003 and 'loss' in model)
        self.assertTrue('loss' not in model.get_managed_params())
        model.save()
        name = model.name

        print('\nsaved, now loading..')

        model = NEModel(
            name=       name,
            loglevel=   10)
        self.assertTrue(model['in_drop']==0.1 and model['in_shape']==784)

    def test_call(self):

        nnm = NEModel(
            nngraph=    fwd_lin_graph,
            in_drop=    0.1)

        inp = np.random.rand(1,784)

        out = nnm.session.run(
            fetches=    nnm['logits'],
            feed_dict=  {nnm['inp_PH']:inp})
        print(out, type(out))
        self.assertTrue(out.shape == (1,10))

    def test_train(self):

        nnm = LinNEModel(
            nngraph=    fwd_lin_graph,
            in_drop=    0.1)

        data = {
            'train': {'inp_PH': np.random.rand(10000,784), 'lbl_PH': np.random.randint(0,9,10000)},
            'test':  {'inp_PH': np.random.rand(1000,784), 'lbl_PH': np.random.randint(0,9,1000)}}

        nnm.load_data(data=data)

        nnm.run_train()

    def test_GX(self):

        # INFO: needs to run saveAB() in a subprocess cause TF elements/graphs do conflict
        @proc_wait
        def saveAB():
            nnm = NEModel(
                name=       'nemodelA',
                nngraph=    fwd_lin_graph,
                in_drop=    0.1)
            nnm.save()
            nnm = NEModel(
                name=       'nemodelB',
                nngraph=    fwd_lin_graph,
                in_drop=    0.1)
            nnm.save()

        saveAB()

        NEModel.gx_saved(
            name_parent_main=   'nemodelA',
            name_parent_scnd=   'nemodelB',
            name_child=         'nemodelC')

        NEModel.gx_saved(
            name_parent_main=   'nemodelA',
            name_parent_scnd=   'nemodelB',
            name_child=         'nemodelD',
            do_gx_ckpt=         False)


if __name__ == '__main__':
    unittest.main()