from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import sys_res_nfo
from pypaq.exception import PyPaqException
from typing import Union, List

DevicesPypaq: Union[None, float, str, List[Union[None,float,str]]] = 1.0


def get_devices(
        devices: DevicesPypaq=  1.0,
        logger=                 None,
        loglevel=               20,
) -> List[None]:
    """ resolves representation given with devices (DevicesTorchness) """

    if not logger:
        logger = get_pylogger(name='get_devices', level=loglevel)

    if type(devices) is not list:
        devices = [devices]

    if not devices:
        raise PyPaqException('no devices given')

    cpu_count = sys_res_nfo()['cpu_count']
    logger.debug(f'got {cpu_count} CPU devices in a system')

    devices_base = []
    for d in devices:

        known_device = False

        if d is None:
            known_device = True
            devices_base.append(d)

        if type(d) is float:
            known_device = True
            if d < 0.0: d = 0.0
            if d > 1.0: d = 1.0
            cpu_count_f = round(cpu_count * d)
            if cpu_count_f < 1: cpu_count_f = 1
            devices_base += [None] * cpu_count_f

        if type(d) is str:
            if d == 'all':
                known_device = True
                devices_base += [None] * cpu_count
            if 'cpu' in d:
                known_device = True
                devices_base.append(None)

        if not known_device:
            msg = f'unknown (not valid?) device given: {d}'
            logger.error(msg)
            raise PyPaqException(msg)

    return devices_base