import os
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

BASE_URL = "https://pakistancode.gov.pk/"
OUTPUT_DIR = "pakistan_laws"

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/140 Safari/537.36"
})

visited = set()
downloaded = set()


def safe_filename(url):
    """Create a stable filename from a URL."""
    path = urlparse(url).path
    name = os.path.basename(path)

    if not name.lower().endswith(".pdf"):
        name += ".pdf"

    # Remove weird filesystem characters
    name = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in name
    )

    return name


def download_pdf(url):
    if url in downloaded:
        return

    try:
        print(f"[PDF] {url}")

        response = session.get(
            url,
            timeout=30,
            stream=True
        )

        response.raise_for_status()

        filename = safe_filename(url)

        # Avoid filename collisions
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            print(f"  -> already exists: {filename}")
            downloaded.add(url)
            return

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"  -> saved: {filepath}")

        downloaded.add(url)

    except Exception as e:
        print(f"  !! failed: {url}")
        print(f"     {e}")


def crawl(url):
    if url in visited:
        return

    visited.add(url)

    try:
        print(f"[PAGE] {url}")

        response = session.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        # If the URL itself is a PDF
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            download_pdf(url)
            return

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for link in soup.find_all("a", href=True):

            href = link["href"]
            absolute_url = urljoin(url, href)

            # PDF
            if (
                ".pdf" in absolute_url.lower()
                or "application/pdf" in href.lower()
            ):
                download_pdf(absolute_url)
                continue

            # Only crawl same domain
            if urlparse(absolute_url).netloc != urlparse(BASE_URL).netloc:
                continue

            # Crawl normal pages
            if absolute_url.startswith(BASE_URL):
                time.sleep(0.3)
                crawl(absolute_url)

    except Exception as e:
        print(f"[ERROR] {url}")
        print(f"        {e}")


if __name__ == "__main__":
    crawl(BASE_URL)

    print("\nDone.")
    print(f"Pages visited: {len(visited)}")
    print(f"PDFs downloaded: {len(downloaded)}")