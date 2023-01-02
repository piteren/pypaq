import random
import time
from typing import Optional, List

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.devices import DevicesParam
from pypaq.mpython.omp import RunningWorker, OMPRunner, ResultOMPRException
from pypaq.scrap.scrap_base import URL, download_response, extract_text, extract_subURLs
from pypaq.textools.text_processing import whitespace_normalization


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


# MultiProcessing Scrapper of List[URL] -> List[UTUS]
class MPScrapper:

    # RunningWorker to retrieve URL content
    class UTUS_retriever(RunningWorker):

        def __init__(self, logger):
            self.logger = logger

        def process(self, url:URL) -> UTUS:
            text = None
            urls = []
            response = download_response(url=url, logger=self.logger)
            if response:
                text = extract_text(response=response, logger=self.logger)
                if text is not None: text = whitespace_normalization(text, remove_nlines=False)
                urls = extract_subURLs(response)
            return UTUS(
                url=    url,
                text=   text,
                urls=   urls)

    def __init__(
            self,
            logger,
            loglevel=               20,
            devices: DevicesParam=  0.5,
            task_timeout=           30,
            report_delay=           5):

        if not logger:
            logger = get_pylogger(
                name=       'MPScrapper',
                add_stamp=  False,
                folder=     None,
                level=      loglevel)
        self.logger = logger

        self.ompr = OMPRunner(
            rw_class=           MPScrapper.UTUS_retriever,
            rw_init_kwargs=     {'logger': logger},
            devices=            devices,
            ordered_results=    False,
            task_timeout=       task_timeout,
            restart_ex_tasks=   False,
            report_delay=       report_delay,
            logger=             self.logger)

        self.logger.info('*** MPScrapper *** initialized')
        self.logger.info(f' > num of workers: {self.ompr.get_num_workers()}')

    def download_UTUSL(
            self,
            urls: List[URL],
            max_time: Optional[int]=    None) -> List[UTUS]:

        self.logger.info(f'MPScrapper is starting to download UTUS for {len(urls)} urls')
        random.shuffle(urls)
        tasks = [{'url': u} for u in urls]
        self.ompr.process(tasks)

        utusL = []
        s_time = time.time()
        for ix in range(len(urls)):
            utusL.append(self.ompr.get_result())
            if max_time is not None:
                tmt = (time.time() - s_time) / 60
                if tmt > max_time:
                    self.logger.warning(f'BREAKING the loop of retrieving RD because max_time exceeded')
                    break

        n_all = len(utusL)
        utusL = [r for r in utusL if type(r) is not ResultOMPRException]  # pop exceptions
        self.logger.info(f'MPScrapper got {len(utusL)} UTUS / {len(urls)} urls ({n_all - len(utusL)} exceptions)')

        return utusL

    def exit(self):
        self.ompr.exit()
