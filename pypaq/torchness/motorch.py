import torch
from torch import nn
from typing import Dict, Optional

from pypaq.lipytools.little_methods import stamp, get_params, get_func_dna
from pypaq.lipytools.logger import set_logger
from pypaq.pms.parasave import ParaSave


# restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
SPEC_KEYS = [
    'name',         # model name
    'train_vars',   # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_vars',     # list of variables returned by opt_func
    'loss',         # loss
    'acc',          # accuracy
    'f1',           # F1
    'batch_size',   # batch size
    'n_batches',    # number of batches for train
    'verb',         # verbosity
]

# defaults below may be given with MOTorch kwargs or overridden by fwd_graph attributes
MOTORCH_DEFAULTS = {
    'seed':         123,                        # seed for torch and numpy
    # TODO:
    #'opt_class':    tf.keras.optimizers.Adam,   # default optimizer of train()
    'devices':      'GPU:0',                    # for TF1 we used '/device:CPU:0'
    # LR management (parameters of LR warmup and annealing)
    'iLR':          3e-4,                       # initial learning rate (base init)
    'warm_up':      None,
    'ann_base':     None,
    'ann_step':     1.0,
    'n_wup_off':    1.0,
    # gradients clipping parameters
    'avt_SVal':     0.1,
    'avt_window':   100,
    'avt_max_upd':  1.5,
    'do_clip':      False,
    # other
    'save_topdir':  '_models',                  # top folder of model save
    'save_fn_pfx':  'todel_dna',           # dna filename prefix
    'load_ckpt':    True,                       # (bool) loads checkpoint (if saved earlier)
    'hpmser_mode':  False,                      # it will set model to be read_only and quiet when running with hpmser
    'read_only':    False,                      # sets model to be read only - wont save anything (wont even create self.model_dir)
    'do_logfile':   True,                       # enables saving log file in self.model_dir
    'do_TB':        True}                       # runs TensorBard, saves in self.model_dir



class MOTorch(ParaSave, nn.Module):

    def __init__(
            self,
            model: type(nn.Module),
            name: Optional[str]=    None,
            name_timestamp=         False,      # adds timestamp to the model name
            save_topdir=            MOTORCH_DEFAULTS['save_topdir'],
            save_fn_pfx=            MOTORCH_DEFAULTS['save_fn_pfx'],
            verb=                   0,
            **kwargs):

        name = model.__class__.__name__ if not name else name
        if name_timestamp: name += f'.{stamp()}'
        if verb>0: print(f'\n *** MOTorch {name} (type: {type(self).__name__}) *** initializes..')

        # ************************************************************************* manage (resolve) DNA & init ParaSave

        # load dna from folder
        dna_saved = ParaSave.load_dna(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        _model_init_params = get_params(model.__init__)
        _model_init_params_defaults = _model_init_params['with_defaults']   # get init params defaults

        dna = {'model': model}
        dna.update(MOTORCH_DEFAULTS)
        dna.update(_model_init_params_defaults)
        dna.update(dna_saved)
        dna.update(kwargs)          # update with kwargs given NOW by user
        dna.update({
            'name':         name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx,
            'verb':         verb})

        ParaSave.__init__(self, lock_managed_params=True, **dna)
        self.check_params_sim(SPEC_KEYS + list(MOTORCH_DEFAULTS.keys())) # safety check

        # hpmser_mode - early override
        if self['hpmser_mode']:
            self.verb = 0
            self['read_only'] = True

        # read only - early override
        if self['read_only']:
            self['do_logfile'] = False
            self['do_TB'] = False

        self.model_dir = f'{self.save_topdir}/{self.name}'
        if self.verb>0: print(f' > MOTorch dir: {self.model_dir}{" read only mode!" if self["read_only"] else ""}')

        if self['do_logfile']:
            set_logger(
                log_folder=     self.model_dir,
                custom_name=    self.name,
                verb=           self.verb)

        dna = self.get_point()
        dna_model_keys = _model_init_params['without_defaults'] + list(_model_init_params['with_defaults'].keys())
        dna_model_keys.remove('self')
        dna_model = {k: dna[k] for k in dna_model_keys}

        if self.verb>0:

            not_used_kwargs = {}
            for k in kwargs:
                if k not in get_func_dna(model.__init__, dna):
                    not_used_kwargs[k] = kwargs[k]

            print(f'\n > MOTorch DNA sources:')
            print(f' >> MOTORCH_DEFAULTS:                  {MOTORCH_DEFAULTS}')
            print(f' >> model init defaults:               {_model_init_params_defaults}')
            print(f' >> DNA saved:                         {dna_saved}')
            print(f' >> given kwargs:                      {kwargs}')
            print(f' >> MOTorch kwargs not used by model : {not_used_kwargs}')
            print(f' MOTorch model DNA:                    {dna_model}')
            print(f' MOTorch complete DNA:                 {dna}')

        # TODO: manage devices

        torch.manual_seed(self['seed'])
        torch.cuda.manual_seed(self['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # https://pytorch.org/docs/stable/notes/randomness.html

        self._torch_module = model
        self._torch_module.__init__(self, **dna_model)

        if self['load_ckpt']:
            try:
                self._load_ckpt()
                if self.verb>0: print(f' > TOModel checkpoint loaded..')
            except Exception as e:
                if self.verb>0: print(f' > TOModel checkpoint NOT loaded ({e})..')

        if self.verb>0: print(f'\n > MOTorch init finished..')

    def forward(self, *args, **kwargs):
        return self._torch_module.forward(self, *args, **kwargs)

    # reloads model checkpoint
    def _load_ckpt(self):
        # TODO: load all that has been saved
        checkpoint = torch.load(f'{self.model_dir}/{self.name}.pt')
        self._torch_module.load_state_dict(self, checkpoint['model_state_dict'])

    # saves model checkpoint
    def _save_ckpt(self):
        # TODO: decide what to save
        torch.save({
            #'epoch': 5,
            'model_state_dict': self._torch_module.state_dict(self),
            # 'optimizer_state_dict': optimizer.state_dict(),
            #'loss': 0.4
        }, f'{self.model_dir}/{self.name}.pt')

    # saves MOTorch (ParaSave DNA and checkpoint)
    def save(self):
        assert not self['read_only'], 'ERR: read only MOTorch cannot be saved!'
        self.save_dna()
        self._save_ckpt()
        if self.verb>0: print(f'MOTorch {self.name} saved')

    def __str__(self):
        return f'{ParaSave.__str__(self)}\n\n{self._torch_module.__str__(self)}'