"""

 2018 (c) piteren

 MultiSaver for NN models
 - implements support for many savers
 - stores (saves and loads) variables in subfolders named with keys of 'vars' dict (default 'ALL')
 - supports list of savers (names for different savers for same variable list)

"""

from typing import Dict, List, Optional, Tuple

from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.neuralmess.get_tf import tf


class MultiSaver:

    def __init__(
            self,
            model_name :str,
            vars: List or Dict[str,List],               # variables may be given in list or dict {key: list} where key is name for the list
            save_TFD :str,                              # save topfolder
            savers: Tuple or List=          (None,),    # tuple of savers names managed by MultiSaver
            max_keep: Optional[List[int]]=  None,       # for each saver keeps N last ckpt saves, for None: 1 for each saver
            session=                        None,
            logger=                         None):
            
        self._log = logger or get_pylogger()

        self.save_FD = f'{save_TFD}/{model_name}'
        self.model_name = model_name

        if type(vars) is list: vars = {'ALL': vars}
        self.vars = vars

        self.session = session

        if max_keep is None: max_keep = [1 for _ in savers]
        assert len(max_keep) == len(savers)

        self.savers = {}
        for ix in range(len(savers)):
            self.savers[savers[ix]] = {}
            for var in self.vars:
                self.savers[savers[ix]][var] = tf.train.Saver(
                    var_list=               self.vars[var],
                    pad_step_number=        True,
                    save_relative_paths=    True,
                    max_to_keep=            max_keep[ix])

        self.s_step = {sv: {var: 0 for var in self.vars} for sv in savers} # self step per saver per vars

        self._log.info(f'*** MultiSaver for {self.model_name} model')
        self._log.debug(f'> MultiSaver folder: {self.save_FD}')
        self._log.debug(f'> gots {len(self.vars)} lists of variables')
        for var in self.vars: self._log.log(5, f' >> {var} - {len(self.vars[var])} variables')
        self._log.debug(f'> for every var list gots {len(savers)} savers: {savers}')

    # saves checkpoint of given saver
    def save(
            self,
            saver=      None,   # saver name
            step :int=  None,   # for None uses self step
            session=    None):

        assert saver in self.savers, 'ERR: unknown saver'

        prep_folder(self.save_FD)
        sv_name = f' {saver}' if saver else ''
        self._log.debug(f'MultiSaver{sv_name} saves variables...')

        for var in self.vars:
            ckpt_path = f'{self.save_FD}/{var}/{self.model_name}'
            if saver: ckpt_path += f'_{saver}'
            ckpt_path += '.ckpt'

            if not session: session = self.session

            latest_filename = 'checkpoint'
            if saver: latest_filename += f'_{saver}'

            self.savers[saver][var].save(
                sess=               session,
                save_path=          ckpt_path,
                global_step=        step if step else self.s_step[saver][var],
                latest_filename=    latest_filename,
                write_meta_graph=   False,
                write_state=        True)
            self.s_step[saver][var] += 1
            self._log.debug(f' > saved variables {var}')

    # loads last checkpoint of given saver
    def load(
            self,
            saver=                          None,
            session :tf.compat.v1.Session=  None,
            allow_init=                     True):

        if not session: session = self.session

        for var in self.vars:
            # look for checkpoint
            latest_filename = 'checkpoint'
            if saver: latest_filename += '_' + saver
            ckpt = tf.train.latest_checkpoint(
                checkpoint_dir=     self.save_FD + '/' + var,
                latest_filename=    latest_filename) if self.save_FD else None

            if ckpt:
                """
                from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
                print_tensors_in_checkpoint_file(
                    file_name=      ckpt,
                    tensor_name=    '',
                    all_tensors=    False)
                """
                self.savers[saver][var].restore(session, ckpt)
                self._log.debug(f'Variables {var} restored from checkpoint {saver if saver else ""}')

            else:
                assert allow_init, 'Err: saver load failed: checkpoint not found and not allowInit'
                session.run(tf.initializers.variables(self.vars[var]))
                self._log.debug(f'No checkpoint found, variables {var} initialized with default initializer')