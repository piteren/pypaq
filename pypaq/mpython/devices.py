import GPUtil
import os
import platform
from typing import Optional, Union, List

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import sys_res_nfo


"""
devices: DevicesParam - (parameter) represents some devices (GPU / CPU)
    # ******************************************************************************************** pypaq representations
    int                 - (int)       single (system) CUDA ID
    -int                - (int)       AVAILABLE CUDA[-int] 
    [] (empty list)     - (list)      all AVAILABLE CUDA
    None                - (NoneType)  single CPU core
    float               - (float)     (0.0;1.0> - factor of system CPU cores
    'all'               - (str)       all CPU cores
    [int,-1,None,'all'] - (list)      list with mix of above, possible repetitions
    # ******************************************************************************** TF1, TF2, PyTorch representations
    '/device:CPU:0'     - (str)       TF1 for CPU
    '/device:GPU:0'     - (str)       TF1 for GPU
    'CPU:0'             - (str)       TF2 for CPU
    'GPU:0'             - (str)       TF2 for GPU
    'cpu'               - (str)       PyTorch for CPU
    'cuda'              - (str)       PyTorch for GPU
    'cuda:0'            - (str)       PyTorch for GPU
    [str]               - (list)      list with strings of above, possible repetitions   
"""
DevicesParam: Union[int, None, float, str, list] = -1


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

# resolves representation given with DevicesParam into middleform (pypaq: List[Union[int,None]]) or TF / PyTorch supported List[str]
def get_devices(
        devices: DevicesParam=      -1,
        namespace: Optional[str]=   None,
        logger=                     None,
        loglevel=                   20) -> List[Union[int,None,str]]:

    if not logger: logger = get_pylogger(level=loglevel)

    if namespace not in [None,'TF1','TF2','torch']:
        msg = 'Wrong namespace, supported are: None, TF1, TF2 or torch'
        logger.error(msg)
        raise NameError(msg)

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
        devices = [None]*num

    pypaq_devices = []
    if devices == []: pypaq_devices = available_cuda_id
    for d in devices:

        known_device = False

        if type(d) is int:
            if d < 0: pypaq_devices.append(available_cuda_id[d])
            else:     pypaq_devices.append(d)
            known_device = True

        if type(d) is float:
            if d < 0.0: d = 0.0
            if d > 1.0: d = 1.0
            cpu_count_f = round(cpu_count * d)
            if cpu_count_f < 1: cpu_count_f = 1
            pypaq_devices += [None]*cpu_count_f
            known_device = True

        if d == []:
            pypaq_devices += available_cuda_id
            known_device = True

        if type(d) is str:
            if d == 'all':
                pypaq_devices += [None]*cpu_count
                known_device = True
            d_low = d.lower()
            if 'cpu' in d_low:
                pypaq_devices.append(None)
                known_device = True
            if 'gpu' in d_low or 'cuda' in d_low:
                dn = 0
                if ':' in d_low: dn = int(d_low.split(':')[-1])
                pypaq_devices.append(dn)
                known_device = True

        if d is None:
            pypaq_devices.append(d)
            known_device = True

        if not known_device:
            msg = f'unknown (not valid?) device given: {d}'
            logger.error(msg)
            raise Exception(msg)

    if namespace is None:
        logger.debug(f'get_devices is returning {len(pypaq_devices)} pypaq_devices: {pypaq_devices}')
        return pypaq_devices

    else:

        device_pfx = '/device:'
        if namespace in ['TF2','torch']: device_pfx = ''

        cpu_pfx = 'cpu' if 'TF' not in namespace else 'CPU'
        gpu_pfx = 'cuda' if 'TF' not in namespace else 'GPU'

        lib_devices = []
        for dev in pypaq_devices:
            if type(dev) is int:
                lib_devices.append(f'{device_pfx}{gpu_pfx}:{dev}')
            if dev is None:
                suffix = '' if namespace == 'torch' else ':0'
                lib_devices.append(f'{device_pfx}{cpu_pfx}{suffix}')

        logger.debug(f'get_devices is returning {len(lib_devices)} lib_devices: {lib_devices}')
        return lib_devices

# masks GPUs from given list of ids or single one
def mask_cuda(ids: Optional[List[int] or int]=  None):
    if ids is None: ids = []
    if type(ids) is int: ids = [ids]
    mask = ''
    for id in ids: mask += f'{int(id)},'
    if len(mask) > 1: mask = mask[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = mask

# wraps mask_cuda to hold DevicesParam
def mask_cuda_devices(devices: DevicesParam=-1, logger=None):
    devices = get_devices(devices, namespace=None, logger=logger)
    ids = [d for d in devices if type(d) is int]
    mask_cuda(ids)