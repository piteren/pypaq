import GPUtil
import os
import platform
from typing import Optional, Union, List

import torch

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import sys_res_nfo


"""
devices: DevicesPypaq - parameter type
    - represents some devices (GPU / CPU)
    - compatible with torch.device (class) type
 
    ### ****************************************************************************************** pypaq representations
    
    # cuda
    int                                 (int)                    single (system) CUDA ID
    -int                                (int)                    AVAILABLE CUDA[-int] 
    [] (empty list)                     (list)                   all AVAILABLE CUDA
    
    # cpu
    None                                (NoneType)               single CPU core
    float                               (float)                  (0.0;1.0> - factor of system CPU cores
    'all'                               (str)                    all CPU cores
    
    ### ***************************************************************************************** PyTorch representation
    'cpu'                               (str)                    PyTorch CPU
    'cuda'                              (str)                    PyTorch GPU
    'cuda:0'                            (str)                    PyTorch GPU
    torch.device                        (<class 'torch.device'>) PyTorch device type
    
    [int,-int,[],None,str,torch.device] (list)                   list with mix of ALL above, possible repetitions
       
"""
DevicesPypaq: Union[int, None, float, str, torch.device, List[Union[int,None,float,str,torch.device]]] = -1


# returns cuda memory size (system first device)
def get_cuda_mem():
    devs = GPUtil.getGPUs()
    if devs: return devs[0].memoryTotal
    else: return 12000 # safety return for no cuda devices case

# returns list of available GPUs ids
def get_available_cuda_id(max_mem=None) -> list: # None sets automatic, otherwise (0,1.1] (above 1 for all)
    if not max_mem:
        tot_mem = get_cuda_mem()
        if tot_mem < 5000:  max_mem=0.35 # small GPU case, probably system single GPU
        else:               max_mem=0.2
    return GPUtil.getAvailable(limit=20, maxMemory=max_mem)

# prints report of system cuda devices
def report_cuda() -> str:
    rp = 'System CUDA devices:'
    for device in GPUtil.getGPUs():
        rp += f'\n > id: {device.id}, name: {device.name}, MEM: {int(device.memoryUsed)}/{int(device.memoryTotal)} (U/T)'
    return rp

# returns dev_pypaq base form (list of int or None)
def to_dev_pypaq_base(
        devices: DevicesPypaq=  -1,
        logger=                 None,
        loglevel=               20,
) -> List[Union[int,None]]:

    if not logger: logger = get_pylogger(level=loglevel)

    if type(devices) is not list: devices = [devices]  # first convert to list

    cpu_count = sys_res_nfo()['cpu_count']

    # look for available CUDA
    available_cuda_id = []
    if platform.system() == 'Darwin': # OSX
        logger.warning('no GPUs available for OSX, using only CPU')
    else:
        available_cuda_id = get_available_cuda_id()

    if not available_cuda_id:
        logger.debug('no GPUs available, using only CPU')
        num = len(devices)
        if num == 0: num = cpu_count
        devices = [None] * num

    devices_base = []

    if devices == []:
        devices_base = available_cuda_id

    for d in devices:

        known_device = False

        if type(d) is int:
            known_device = True
            if d < 0: devices_base.append(available_cuda_id[d])
            else:     devices_base.append(d)

        if d == []:
            known_device = True
            devices_base += available_cuda_id

        if d is None:
            known_device = True
            devices_base.append(d)

        if type(d) is float:
            known_device = True
            if d < 0.0: d = 0.0
            if d > 1.0: d = 1.0
            cpu_count_f = round(cpu_count * d)
            if cpu_count_f < 1: cpu_count_f = 1
            devices_base += [None]*cpu_count_f

        if type(d) is torch.device:
            d = str(d)

        if type(d) is str:
            if d == 'all':
                known_device = True
                devices_base += [None]*cpu_count
            if 'cpu' in d:
                known_device = True
                devices_base.append(None)
            if 'cuda' in d:
                known_device = True
                dn = 0
                if ':' in d:
                    dn = int(d.split(':')[-1])
                devices_base.append(dn)

        if not known_device:
            msg = f'unknown (not valid?) device given: {d}'
            logger.error(msg)
            raise Exception(msg)

    return devices_base

# resolves representation given with DevicesPypaq into dev_pypaq base form or List[str] in torch accepted namespace
def get_devices(
        devices: DevicesPypaq=  -1,
        torch_namespace: bool=  True,
        logger=                 None,
        loglevel=               20,
) -> List[Union[int,None,str]]:

    if not logger: logger = get_pylogger(level=loglevel)

    devices_base = to_dev_pypaq_base(devices=devices, logger=logger)

    if not torch_namespace:
        return devices_base
    else:
        return [f'cuda:{dev}' if type(dev) is int else 'cpu' for dev in devices_base]


# masks GPUs from given list of ids or single one
def mask_cuda(ids: Optional[List[int] or int]=  None):
    if ids is None: ids = []
    if type(ids) is int: ids = [ids]
    mask = ''
    for id in ids: mask += f'{int(id)},'
    if len(mask) > 1: mask = mask[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = mask

# wraps mask_cuda to hold DevicesPypaq
def mask_cuda_devices(
        devices: DevicesPypaq=  -1,
        logger=                 None):
    devices = get_devices(devices, torch_namespace=False, logger=logger)
    ids = [d for d in devices if type(d) is int]
    mask_cuda(ids)