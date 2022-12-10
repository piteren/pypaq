import os
import warnings
from typing import Optional

from pypaq.lipytools.logger import set_logger
from pypaq.mpython.devices import DevicesParam, get_devices
from pypaq.neuralmess.get_tf import tf

# init function for every TF.py script:
# - sets low verbosity of TF
# - starts logger
# - manages TF devices
def nestarter(
        log_folder: Optional[str]=  '_log', # for None doesn't log
        custom_name: str or None=   None,   # set custom logger name, for None uses default
        devices: DevicesParam=      -1,     # False to not manage TF devices
        verb=                       1,
        silent_error=               False): # turns off any TF errors, be careful

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' if silent_error else '2'

    if log_folder: set_logger(log_folder=log_folder, custom_name=custom_name, verb=verb) # set logger
    if devices is not False: return get_devices(devices=devices, namespace='TF1')