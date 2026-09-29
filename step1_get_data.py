"""
Step 1: Download 10-K filings from SEC EDGAR for 3 companies and clean the text.
Output: raw_filings/<TICKER>.txt  (clean plain text, ready for chunking in Step 2)
"""

import requests
from bs4 import BeautifulSoup
import os
import time

HEADERS = {"User-Agent": "Piggy Student Project piggy@example.com"}

COMPANIES = {
    "AAPL": "0000320193",
    "TSLA": "0001318605",
    "MSFT": "0000789019",
}

os.makedirs("raw_filings", exist_ok=True)


def get_latest_10k_url(cik: str) -> str:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()

    recent = data["filings"]["recent"]
    for form, accession, doc in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "10-K":
            accession_nodash = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_nodash}/{doc}"
            )
    raise ValueError("No 10-K found")


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "")):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text


def main():
    for ticker, cik in COMPANIES.items():
        print(f"[{ticker}] finding latest 10-K...")
        doc_url = get_latest_10k_url(cik)

        print(f"[{ticker}] downloading {doc_url}")
        html = requests.get(doc_url, headers=HEADERS).text

        print(f"[{ticker}] cleaning...")
        text = clean_html_to_text(html)

        out_path = f"raw_filings/{ticker}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[{ticker}] saved {len(text):,} chars -> {out_path}\n")
        time.sleep(0.5)


if __name__ == "__main__":
    main()