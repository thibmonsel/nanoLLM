import os
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrape_wikipedia_article(url):
    """Scrapes a Wikipedia article and returns its title and content."""
    # Add a random delay to avoid overloading Wikipedia servers
    time.sleep(random.uniform(1, 3))
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch {url}: Status code {response.status_code}")
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")
    # Get the article title
    title = soup.find(id="firstHeading").text
    # Get the article content
    content_div = soup.find(id="mw-content-text")
    # Remove unwanted elements
    for unwanted in content_div.select(".mw-editsection, .reference, .reflist, table"):
        unwanted.extract()

    # Get paragraphs
    paragraphs = [p.text for p in content_div.find_all("p")]
    content = "\n\n".join(paragraphs)
    return title, content


def scrape_multiple_articles(urls):
    """Scrapes multiple Wikipedia articles and returns them as a DataFrame."""
    results = []

    for url in urls:
        print(f"Scraping: {url}")
        title, content = scrape_wikipedia_article(url)
        if title and content:
            results.append({"url": url, "title": title, "content": content})

    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # List of Wikipedia URLs to scrape
    wiki_urls = [
        "https://en.wikipedia.org/wiki/Napoleon",
        "https://en.wikipedia.org/wiki/Louis_XVII",
        "https://en.wikipedia.org/wiki/Louis_XVI",
        "https://en.wikipedia.org/wiki/Louis_XV",
        "https://en.wikipedia.org/wiki/Louis_XIII",
        "https://en.wikipedia.org/wiki/Henry_IV_of_France",
        "https://en.wikipedia.org/wiki/Henry_III_of_France",
        "https://en.wikipedia.org/wiki/Charles_IX_of_France",
        "https://en.wikipedia.org/wiki/Francis_II_of_France",
        "https://en.wikipedia.org/wiki/Francis_I_of_France",
        "https://en.wikipedia.org/wiki/Louis_XII",
        "https://en.wikipedia.org/wiki/Charles_VIII_of_France",
        "https://en.wikipedia.org/wiki/Louis_XI",
        "https://en.wikipedia.org/wiki/Charles_VII_of_France",
        "https://en.wikipedia.org/wiki/Charles_VI_of_France",
        "https://en.wikipedia.org/wiki/Charles_V_of_France",
        "https://en.wikipedia.org/wiki/John_II_of_France",
        "https://en.wikipedia.org/wiki/Philip_VI_of_France",
        "https://en.wikipedia.org/wiki/Charles_IV_of_France",
        "https://en.wikipedia.org/wiki/Philip_V_of_France",
        "https://en.wikipedia.org/wiki/John_I_of_France",
        "https://en.wikipedia.org/wiki/Louis_X_of_France",
        "https://en.wikipedia.org/wiki/Philip_IV_of_France",
        "https://en.wikipedia.org/wiki/Philip_III_of_France",
        "https://en.wikipedia.org/wiki/Louis_IX_of_France",
        "https://en.wikipedia.org/wiki/Louis_VIII_of_France"
        "https://en.wikipedia.org/wiki/Philip_II_of_France",
        "https://en.wikipedia.org/wiki/Louis_VII_of_France",
        "https://en.wikipedia.org/wiki/Louis_VI_of_France",
        "https://en.wikipedia.org/wiki/Philip_I_of_France",
        "https://en.wikipedia.org/wiki/Henry_I_of_France",
        "https://en.wikipedia.org/wiki/Robert_II_of_France",
        "https://en.wikipedia.org/wiki/Hugh_Capet",
    ]

    # Scrape articles
    articles_df = scrape_multiple_articles(wiki_urls)

    new_dir = os.path.dirname(__file__)
    os.makedirs(new_dir, exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "kings.txt"), "w") as text_file:
        for content in articles_df["content"]:
            text_file.write(content)
