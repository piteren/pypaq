"""

 2019 (c) piteren

"""

import GPUtil
import os
import platform
from typing import List, Optional
import warnings

from pypaq.lipytools.logger import set_logger
from pypaq.mpython.mptools import DevicesParam
from pypaq.neuralmess.get_tf import tf


# masks GPUs from given list of ids or single one
def mask_cuda(ids :Optional[List[int] or int]=  None):
    if ids is None: ids = []
    if type(ids) is int: ids = [ids]
    mask = ''
    for id in ids: mask += f'{int(id)},'
    if len(mask) > 1: mask = mask[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = mask

# wraps mask_cuda to hold DevicesParam
def mask_cuda_devices(devices :DevicesParam=-1, verb=0):

    devices = tf_devices(devices, verb=verb)

    ids = []
    devices_other = []
    devices_gpu = []
    for device in devices:
        if 'GPU' in device: devices_gpu.append(device)
        else: devices_other.append(device)
    if devices_gpu: ids = [dev[12:] for dev in devices_gpu]
    mask_cuda(ids)

# returns cuda memory size (system first device)
def get_cuda_mem():
    devs = GPUtil.getGPUs()
    if devs: return devs[0].memoryTotal
    else: return 12000 # safety return for no cuda devices case

# returns list of available GPUs ids
def get_available_cuda_id(max_mem=None): # None sets automatic, otherwise (0,1.1] (above 1 for all)
    if not max_mem:
        tot_mem = get_cuda_mem()
        if tot_mem < 5000:  max_mem=0.35 # small GPU case, probably system single GPU
        else:               max_mem=0.2
    return GPUtil.getAvailable(limit=20, maxMemory=max_mem)

# prints report of system cuda devices
def report_cuda():
    print('\nSystem CUDA devices:')
    for device in GPUtil.getGPUs():
        print(f' > id: {device.id}, name: {device.name}, MEM: {int(device.memoryUsed)}/{int(device.memoryTotal)} (U/T)')

# resolves given devices, returns list of str in TF naming convention, for OSX returns CPU only
def tf_devices(
        devices :DevicesParam=  -1,
        verb=                   1) -> List[str]:

    if type(devices) is not list: devices = [devices]  # first convert to list

    force_CPU = False
    # OSX
    if platform.system() == 'Darwin':
        print('no GPUs available for OSX, using only CPU')
        force_CPU = True
    # no GPU @system
    if not force_CPU and not GPUtil.getGPUs():
        print('no GPUs available, using only CPU')
        force_CPU = True
    if force_CPU:
        num = len(devices)
        if not num: num = 1
        devices = [None] * num

    # [] or -1 case >> check available GPU and replace with positive ints
    if not devices or -1 in devices:
        av_dev = get_available_cuda_id()
        if not av_dev: raise Exception('No available GPUs!')
        if not devices: devices = av_dev
        else:
            pos_devices = []
            for dev in devices:
                if dev == -1: pos_devices.append(av_dev[-1])
                else:         pos_devices.append(dev)
            devices = pos_devices

    # split devices into 3 lists
    devices_str = []
    devices_int = []
    devices_CPU = []
    for dev in devices:
        if type(dev) is str: devices_str.append(dev)
        if type(dev) is int: devices_int.append(dev)
        if dev is None: devices_CPU.append(None)

    # prepare final list
    final_devices = []
    final_devices += devices_str # just add str
    final_devices += ['/device:CPU:0'] * len(devices_CPU)
    if devices_int:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if verb>0: report_cuda()
        for dev in devices_int: final_devices.append(f'/device:GPU:{dev}')

    if verb>0: print(f' > returning {len(final_devices)} devices: {final_devices}')
    return final_devices

# init function for every TF.py script:
# - sets low verbosity of TF
# - starts logger
# - manages TF devices
def nestarter(
        log_folder: Optional[str]=  '_log', # for None doesn't log
        custom_name: str or None=   None,   # set custom logger name, for None uses default
        devices=                    -1,     # False to not manage TF devices
        verb=                       1,
        silent_error=               False): # turns off any TF errors, be careful

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' if silent_error else '2'

    if log_folder: set_logger(log_folder=log_folder, custom_name=custom_name, verb=verb) # set logger
    if devices is not False: return tf_devices(devices=devices, verb=verb)


if __name__ == '__main__':

    devices_conf = [
        0,
        1,
        -1,
        None,
        '/device:GPU:0',
        ['/device:GPU:0','/device:GPU:0'],
        [],
        [0,1],
        [0,0],
        [0,-1],
        [-1,-1],
        [None]*5,
        [None,0,1,-1,'/device:GPU:0',None]]

    for dc in devices_conf:
        print(f'\n ### DC: {dc}')
        print(tf_devices(devices=dc, verb=1))

    #print(get_cuda_mem())
