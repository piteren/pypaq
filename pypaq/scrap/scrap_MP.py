import random
import time
from typing import Optional, List, Dict

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.devices import DevicesParam
from pypaq.mpython.omp import RunningWorker, OMPRunner
from pypaq.scrap.scrap_base import URL, download_response


# Url, Text, sub-UrlS - this class keeps URL content (URL, its text and its sub-urls (links))
class UTUS:
    def __init__(
            self,
            url: URL,
            text: Optional[str],  # url text
            urls: List[URL]  # url sub-links
    ):
        self.url = url
        self.text = text
        self.urls = urls


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
            restart_ex_tasks=   False,
            report_delay=       report_delay,
            logger=             self.logger,
            **kwargs)

        self.headers = headers
        self.proxies = proxies

        self.logger.info('*** MPScrapper *** initialized')
        self.logger.info(f' > num of workers: {self.get_num_workers()}')

    # wraps process() + get_all_results() with tasks preparation, with header & proxy management, timeout
    def scrap(
            self,
            urls: List[URL],
            max_time: Optional[int]=    None,   # max num minutes for download()
            retry: Optional[int]=       2,      # number of retries for 429
    ) -> List[dict]:

        self.logger.info(f'MPScrapper is starting to download RESPONSES for {len(urls)} urls')
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
            sc_results.append(self.get_result())
            if max_time is not None:
                tmt = (time.time() - s_time) / 60
                if tmt > max_time:
                    self.logger.warning(f'BREAKING the loop of download_RESPONSES because max_time exceeded')
                    break
        self.logger.info(f'MPScrapper got {len(sc_results)} sc_results, time taken: {(time.time() - s_time) / 60:.1f} min')

        return sc_results