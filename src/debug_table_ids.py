import requests
from bs4 import BeautifulSoup, Comment

SEASON = 2025
URL = f"https://www.sports-reference.com/cbb/schools/syracuse/{SEASON}.html"


def extract_table_ids(html: str):
    soup = BeautifulSoup(html, "html.parser")

    ids = set()

    # normal tables
    for t in soup.find_all("table"):
        if t.get("id"):
            ids.add(t["id"])

    # tables inside HTML comments
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        if "table" in c:
            csoup = BeautifulSoup(c, "html.parser")
            for t in csoup.find_all("table"):
                if t.get("id"):
                    ids.add(t["id"])

    return sorted(ids)


if __name__ == "__main__":
    r = requests.get(URL, timeout=30)
    r.raise_for_status()

    table_ids = extract_table_ids(r.text)
    print(f"Found {len(table_ids)} table ids on {URL}:\n")
    for tid in table_ids:
        print(tid)
