from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import sys_res_nfo
from pypaq.exception import PyPaqException

DevicesPypaq = None | float | str | list[None | float | str]


def get_devices(
        devices: DevicesPypaq = 1.0,
        logger = None,
        loglevel = 20,
) -> list[None]:
    """ resolves representation given with devices """

    if not logger:
        logger = get_pylogger(name='get_devices', level=loglevel)

    devices_list = devices if isinstance(devices, list) else [devices]

    if not devices_list:
        raise PyPaqException('no devices given')

    cpu_count = sys_res_nfo()['cpu_count']
    logger.debug(f'got {cpu_count} CPU devices in a system')

    devices_base = []
    for d in devices_list:

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