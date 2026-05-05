import logging

from multiprocessing import cpu_count
from pypaq.exception import PyPaqException

DevicesPypaq = None | float | str | list[None | float | str]

logger = logging.getLogger(__name__)


def get_devices(devices: DevicesPypaq = 1.0) -> list[None]:
    """ resolves representation given with devices """

    devices_list = devices if isinstance(devices, list) else [devices]

    if not devices_list:
        raise PyPaqException('no devices given')

    n_cpus = cpu_count()
    logger.debug(f'got {n_cpus} CPU devices in a system')

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
            n_cpus_f = round(n_cpus * d)
            if n_cpus_f < 1: n_cpus_f = 1
            devices_base += [None] * n_cpus_f

        if type(d) is str:
            if d == 'all':
                known_device = True
                devices_base += [None] * n_cpus
            if 'cpu' in d:
                known_device = True
                devices_base.append(None)

        if not known_device:
            msg = f'unknown (not valid?) device given: {d}'
            logger.error(msg)
            raise PyPaqException(msg)

    return devices_base