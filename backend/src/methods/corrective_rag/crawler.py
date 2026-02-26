from langchain_community.tools import DuckDuckGoSearchResults
from crawl4ai import AsyncWebCrawler, CacheMode
import asyncio
from bs4 import BeautifulSoup

class web_search:

    def __init__(self, max_requests):
        self.max_request = max_requests
        self.browser = DuckDuckGoSearchResults(num_results=self.max_request, output_format='list')

    async def __get_website_content__(self, urls):
        async with AsyncWebCrawler(browser_type='chromium', headless=True, verbose=True) as crawler:
            results = await crawler.arun_many(urls=urls, bypass_cache=True, cache_mode=CacheMode.DISABLED, exclude_external_links=True, remove_overlay_elements=True, simulate_user=True, magic=True)
        results = list(filter(lambda result: result.success, results))
        for i in range(len(results)):
            soup = BeautifulSoup(results[i].cleaned_html, 'html.parser')
            texte = soup.get_text()
            results[i] = vars(results[i])
            results[i]['soup_html'] = texte
        return results

    async def __run_search__(self, query):
        search_results = self.browser.run(query)
        urls = [result['link'] for result in search_results]
        results_content = await self.__get_website_content__(urls)
        return results_content

    def search(self, query):
        web_results = asyncio.run(self.__run_search__(query))
        return web_results