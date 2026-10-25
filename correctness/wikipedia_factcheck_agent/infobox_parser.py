# pip install requests beautifulsoup4
from __future__ import annotations
import os
import re
from functools import lru_cache
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup

class WikiInfobox:
    def __init__(self, lang: str = "en"):
        self.base = f"https://{lang}.wikipedia.org/w/api.php"
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": os.environ.get(
                "SA_WIKIMEDIA_USER_AGENT",
                "selective-abstraction-reproducibility/1.0",
            )
        })

    @lru_cache(maxsize=256)
    def fetch_infobox_html(self, title: str) -> Optional[str]:
        """Fetch lead HTML (contains infobox) with redirects + CORS-friendly origin."""
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "redirects": 1,
            "origin": "*",
            "disableeditsection": 1,
            "disablelimitreport": 1,
        }
        r = self.sess.get(self.base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("parse", {}).get("text", {}).get("*")

    @staticmethod
    def _clean_text(s: str) -> str:
        # Collapse whitespace and strip citation footnotes like [1], [2]
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\[\d+\]", "", s)
        return s

    def parse_infobox(self, html: str) -> Dict[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        ibox = soup.select_one("table.infobox")
        if not ibox:
            return {}

        out: Dict[str, str] = {}
        # Walk rows; header rows (th only) usually denote section titles; keep as keys with empty value
        for row in ibox.select("tr"):
            th = row.find("th")
            td = row.find("td")

            # Skip image-only rows
            if td and td.find("img"):
                # Optional: expose main image URL under a special key
                img = td.find("img")
                if img and img.get("src"):
                    out.setdefault("image", "https:" + img["src"] if img["src"].startswith("//") else img["src"])
                continue

            if th and td:
                key = self._clean_text(th.get_text(" ", strip=True))
                val = self._clean_text(td.get_text(" ", strip=True))
                if key and val:
                    out[key] = val
            elif th and not td:
                # Section header within infobox
                key = self._clean_text(th.get_text(" ", strip=True))
                if key:
                    out.setdefault(f"__section__:{key}", "")
        return out

    def get_infobox(self, title: str) -> Dict[str, str]:
        html = self.fetch_infobox_html(title)
        if not html:
            return {}
        return self.parse_infobox(html)

# ---- Example usage ----
if __name__ == "__main__":
    wiki = WikiInfobox(lang="en")
    data = wiki.get_infobox("Don Featherstone (artist)")
    for k, v in list(data.items())[:15]:
        print(f"{k}: {v}")
