import time
from typing import List, Dict
import urllib

from pypaq.scrap.scrap_base import URL, download_response, extract_subURLs

GOOGLE_DOMAINS = (
    'https://www.google.',
    'https://google.',
    'https://webcache.googleusercontent.',
    'http://webcache.googleusercontent.',
    'https://policies.google.',
    'https://support.google.',
    'https://maps.google.')

# keeps URL meta-information
class URLMeta:

    def __init__(
            self,
            url: URL,
            gdepth: int,                # depth from google SERP
            from_queries: List[str],    # if gdepth == 0 here goes query
            from_urls: List[URL]        # if gdepth > 0 here goes parent url
    ):
        self.url = url
        self.gdepth = gdepth
        self.from_queries = from_queries
        self.from_urls = from_urls

    # BASELINE, the smaller rank the "better" is URL
    def get_rank(self) -> float:
        rank = self.gdepth
        if self.from_queries: rank -= 0.1 * len(self.from_queries)
        if self.from_urls: rank -= 0.07 * len(self.from_urls)
        return float(rank)


# gets List[URL] from google search page for given query
# https://practicaldatascience.co.uk/data-science/how-to-scrape-google-search-results-using-python
def _dogoogle_urls(query:str, logger) -> List[URL]:

    google_url = f'https://www.google.co.uk/search?q={urllib.parse.quote_plus(query)}'
    google_response = download_response(google_url, logger)

    urls = []
    if not google_response:
        logger.warning(f'query: >{query}< did not return response from google')
        return urls

    urls = extract_subURLs(google_response)
    for url in urls[:]:
        if url.startswith(GOOGLE_DOMAINS):
            urls.remove(url)
    return urls


def download_google_urls(queries:List[str], logger) -> Dict[URL, URLMeta]:
    umD: Dict[URL, URLMeta] = {}
    for q in queries:
        s_time = time.time()
        urls = _dogoogle_urls(q, logger)

        for url in urls:
            if url not in umD:
                umD[url] = URLMeta(
                    url=            url,
                    gdepth=         0,
                    from_queries=   [],
                    from_urls=      [])
            umD[url].from_queries.append(q) # some urls may come from many queries

        logger.info(f'received {len(urls)} urls ({time.time() - s_time:.1f}s) for query: "{q}"')

    logger.info(f'scrapped {len(umD)} unique urls')
    return umD
