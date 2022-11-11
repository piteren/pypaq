import GPUtil
import os
import platform
from typing import Optional, Union, List

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import sys_res_nfo


"""
devices: DevicesParam - (parameter) manages GPUs, gives options for CPUs
    int                 - one (system) CUDA ID
    -1                  - last AVAILABLE CUDA
    'all'               - all CPU cores
    None                - single CPU core
    str_TF_format       - device in TF format ('GPU:0')
    [] (empty list)     - all AVAILABLE CUDA
    [int,-1,None,str]   - list of devices: ints (CUDA IDs), may contain None, possible repetitions
"""
DevicesParam: Union[int, None, str, list] = -1


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
def report_cuda() -> str:
    rp = 'System CUDA devices:'
    for device in GPUtil.getGPUs():
        rp += f'\n > id: {device.id}, name: {device.name}, MEM: {int(device.memoryUsed)}/{int(device.memoryTotal)} (U/T)'
    return rp


def get_devices(
        devices: DevicesParam=  -1,
        namespace: str=         'TF1',
        logger=                  None) -> List[str]:

    if not logger: logger = get_pylogger(name='get_devices')

    if namespace not in ['TF1','TF2','torch']:
        raise NameError('Wrong namespace, supported are: TF1, TF2 or torch')

    device_pfx = '/device:'
    if namespace in ['TF2','torch']: device_pfx = ''

    cpu_pfx = 'cpu' if 'TF' not in namespace else 'CPU'
    gpu_pfx = 'cuda' if 'TF' not in namespace else 'GPU'

    # all CPU case
    if devices == 'all': devices = [None] * sys_res_nfo()['cpu_count']

    if type(devices) is not list: devices = [devices]  # first convert to list

    force_CPU = False
    # OSX
    if platform.system() == 'Darwin':
        print('no GPUs available for OSX, using only CPU')
        force_CPU = True
    # no GPU @system
    if not force_CPU and not get_available_cuda_id():
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

    # reduce str to int
    for dev in devices_str:
        devices_int.append(int(dev.split(':')[-1]))

    # prepare final list
    final_devices = []
    final_devices += [f'{device_pfx}{cpu_pfx}:0'] * len(devices_CPU)
    if devices_int:
        logger.trace(report_cuda())
        for dev in devices_int: final_devices.append(f'{device_pfx}{gpu_pfx}:{dev}')

    logger.debug(f'get_devices is returning {len(final_devices)} devices: {final_devices}')
    return final_devices

# masks GPUs from given list of ids or single one
def mask_cuda(ids: Optional[List[int] or int]=  None):
    if ids is None: ids = []
    if type(ids) is int: ids = [ids]
    mask = ''
    for id in ids: mask += f'{int(id)},'
    if len(mask) > 1: mask = mask[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = mask

# wraps mask_cuda to hold DevicesParam
def mask_cuda_devices(devices: DevicesParam=-1, verb=0):

    devices = get_devices(devices, verb=verb)

    ids = []
    devices_other = []
    devices_gpu = []
    for device in devices:
        if 'GPU' in device: devices_gpu.append(device)
        else: devices_other.append(device)
    if devices_gpu: ids = [dev[12:] for dev in devices_gpu]
    mask_cuda(ids)