import random
import time
from typing import Optional, List, Dict

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.devices import DevicesParam
from pypaq.mpython.ompr import RunningWorker, OMPRunner
from pypaq.scrap.scrap_base import URL, download_response

RESPONSE_CODES = {
    'response.200': 'OK',
    'response.202': 'accepted',
    'response.400': 'bad request',
    'response.401': 'unauthorized',
    'response.403': 'forbidden',
    'response.404': 'not found',
    'response.406': 'not acceptable',
    'response.409': 'conflict',
    'response.410': 'gone',
    'response.429': 'too many requests',
    'response.451': 'unavailable for legal reasons',
    'response.500': 'internal server error',
    'response.502': 'bad gateway',
    'response.503': 'service unavailable',
    'response.521': 'web server is down',
    'response.526': 'invalid SSL certificate',
    'response.999': 'request denied',
}


# RunningWorker to retrieve URL content
class ResponseRetriever(RunningWorker):

    def __init__(self, logger):
        self.logger = logger

    def process(
            self,
            url: URL,
            header: Optional[Dict]= None,
            proxy: Optional[str]=   None) -> dict:

        response = download_response(
            url=    url,
            header= header,
            proxy=  proxy,
            logger= self.logger)

        return {
            'url':      url,
            'header':   header,
            'proxy':    proxy,
            'response': response}

# MultiProcessing Scrapper
class MPScrapper(OMPRunner):

    def __init__(
            self,
            rw_class=                                   ResponseRetriever,
            devices: DevicesParam=                      [None]*4,
            task_timeout=                               30,
            report_delay=                               5,
            headers: Optional[List[Optional[Dict]]]=    None,
            proxies: Optional[List[str]]=               None,
            logger=                                     None,
            loglevel=                                   20,
            **kwargs):

        if not logger:
            logger = get_pylogger(
                name=       self.omp_name,
                add_stamp=  False,
                folder=     None,
                level=      loglevel)
        self.logger = logger

        OMPRunner.__init__(
            self,
            rw_class=           rw_class,
            devices=            devices,
            ordered_results=    False,
            task_timeout=       task_timeout,
            report_delay=       report_delay,
            logger=             self.logger,
            **kwargs)

        self.headers = headers
        self.proxies = proxies

        self.logger.info('*** MPScrapper *** initialized')
        self.logger.info(f'> num of workers: {self.get_num_workers()}')

        self.scrap_stats = {'download_exceptions':0, 'response.None':0}

    # logs global or given scrap stats
    def log_scrap_stats(self, scrap_stats:Optional=None):

        if not scrap_stats:
            scrap_stats = self.scrap_stats

        n_all = sum([scrap_stats[k] for k in scrap_stats])
        self.logger.info('Scrap stats:')
        for k in sorted(list(scrap_stats.keys())):
            nfo = k
            if k in RESPONSE_CODES: nfo = f'{k} - {RESPONSE_CODES[k]}'
            self.logger.info(f'>> {nfo:60} : {scrap_stats[k]:10} {scrap_stats[k]/n_all*100:5.1f}%')
        k = 'ALL'
        self.logger.info(f'>> {k:60} : {n_all:10}')

    # wraps process() + get_all_results() with tasks preparation, with header & proxy management, timeout
    def scrap(
            self,
            urls: List[URL],
            max_time: Optional[int]=    None,   # max num minutes for download()
            #TODO: monitor responses codes, retry blocked with proxies
            retry: Optional[int]=       2,      # number of retries for 429
    ) -> List[dict]:

        self.logger.info(f'MPScrapper is starting to download RESPONSES for {len(urls)} urls')

        scrap_stats = {'download_exceptions': 0, 'response.None': 0} # session scrap stats

        random.shuffle(urls)
        tasks = [{
            'url':      u,
            'header':   random.choice(self.headers) if self.headers else None,
            'proxy':    random.choice(self.proxies) if self.proxies else None,
        } for u in urls]
        self.process(tasks)

        sc_results = []
        s_time = time.time()
        for ix in range(len(urls)):

            result = self.get_result()

            # update stats
            if type(result) is not dict:
                scrap_stats['download_exceptions'] += 1
            else:
                response = result['response']
                if response is None:
                    scrap_stats['response.None'] += 1
                else:
                    sct = f'response.{response.status_code}'
                    if sct not in scrap_stats:
                        scrap_stats[sct] = 0
                    scrap_stats[sct] += 1

            sc_results.append(result)
            if max_time is not None:
                tmt = (time.time() - s_time) / 60
                if tmt > max_time:
                    self.logger.warning(f'BREAKING the loop of download_RESPONSES because max_time exceeded')
                    break
        self.logger.info(f'MPScrapper got {len(sc_results)} sc_results, time taken: {(time.time() - s_time) / 60:.1f} min')
        self.log_scrap_stats(scrap_stats)

        # update global scrap_stats
        for k in scrap_stats:
            if k not in self.scrap_stats:
                self.scrap_stats[k] = 0
            self.scrap_stats[k] += scrap_stats[k]

        return sc_results

    def exit(self):
        self.log_scrap_stats()
        super().exit()