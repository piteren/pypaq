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


# tries to download response (source (HTML) code of URL)
def download_response(url:URL, logger) -> Optional[RESPONSE]:
    for header in HEADERS:
        try:
            session = HTMLSession()
            response = session.get(url, headers=header)
            return response
        except Exception as e:
            msg = f'get_response() got exception: "{e}", url: {url}, header: {header}'
            logger.warning(msg)
            return None

# extracts sub-urls from response
def extract_subURLs(response: RESPONSE) -> List[URL]:
    return list(response.html.absolute_links)

# extracts text from response
def extract_text(response:RESPONSE, logger) -> Optional[str]:
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator='\n')
        #return[data.get_text() for data in soup.find_all("p")]
    except Exception as e:
        msg = f'get_texts() got exception: "{e}"'
        logger.warning(msg)
        return None