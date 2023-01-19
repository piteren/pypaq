from bs4 import BeautifulSoup
import requests
from requests_html import HTMLSession
from typing import Optional, List

RESPONSE = requests.models.Response
URL = str

HEADERS = [
    None, # google does not like header (?)
    {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'},
    {
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
    },
]


# tries to download response (HTML code of URL)
def download_response(url:URL, logger, proxy=None) -> Optional[RESPONSE]:
    try:
        response = None
        for header in HEADERS:
            session = HTMLSession()
            proxies = {'http': f'http://{proxy}'} if proxy else None
            response = session.get(url, headers=header, proxies=proxies)
            if not response:
                logger.warning(f'download_response() received response: {response} for {url}')
            if response: break
        return response
    except Exception as e:
        msg = f'download_response() got exception: "{e}", url: {url}, header: {header}'
        logger.warning(msg)
        return None

# extracts sub-urls from response
def extract_subURLs(response: RESPONSE) -> List[URL]:
    return list(response.html.absolute_links)

# extracts text from response
def extract_text(
        response:RESPONSE,
        logger,
        separator=      '\n',
        encode_decode=  True,
) -> Optional[str]:
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=separator)
        if encode_decode:
            text = text.encode("ascii", "ignore")
            text = text.decode()
        return text
    except Exception as e:
        msg = f'get_texts() got exception: "{e}"'
        logger.warning(msg)
        return None