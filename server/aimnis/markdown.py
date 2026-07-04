"""HTML → Markdown for agents (content negotiation on `Accept: text/markdown`).

Agents that ask for `text/markdown` get a markdown rendering of any HTML page —
the same convention Cloudflare's "Markdown for Agents" uses — at a fraction of
the token cost, while browsers keep getting HTML (see the middleware in api.py).
This is a purpose-built converter for this app's own server-rendered markup via
the stdlib HTMLParser, not a general-purpose one: it covers the tags our pages
use (headings, links, emphasis, code/pre, lists, tables-as-text) and drops
non-content elements (style, script, inline SVG charts).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_DROP = {"style", "script", "svg", "head", "template"}  # subtree contributes nothing
_BLOCK = {"p", "div", "section", "article", "footer", "nav", "form", "table",
          "tr", "ul", "ol", "blockquote", "details"}


class _Converter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._drop = 0        # depth inside dropped subtrees
        self._pre = 0         # depth inside <pre> (whitespace is significant)
        self._href: str | None = None
        self._link_text: list[str] = []

    # -- emit helpers -------------------------------------------------------- #
    def _emit(self, text: str) -> None:
        (self._link_text if self._href is not None else self.out).append(text)

    def _block_break(self) -> None:
        self._emit("\n\n")

    # -- parser callbacks ---------------------------------------------------- #
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP:
            self._drop += 1
            return
        if self._drop:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._href, self._link_text = href, []
        elif tag in ("b", "strong"):
            self._emit("**")
        elif tag in ("i", "em"):
            self._emit("*")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "pre":
            self._pre += 1
            self._emit("\n\n```\n")
        elif tag == "li":
            self._emit("\n- ")
        elif tag in ("br", "hr"):
            self._emit("\n")
        elif tag in ("td", "th"):
            self._emit(" · ")
        elif tag in _BLOCK:
            self._block_break()

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP:
            self._drop = max(0, self._drop - 1)
            return
        if self._drop:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._block_break()
        elif tag == "a" and self._href is not None:
            text = "".join(self._link_text).strip() or self._href
            href, self._href = self._href, None
            self.out.append(f"[{text}]({href})")
        elif tag in ("b", "strong"):
            self._emit("**")
        elif tag in ("i", "em"):
            self._emit("*")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._emit("\n```\n\n")
        elif tag in _BLOCK:
            self._block_break()

    def handle_data(self, data: str) -> None:
        if self._drop:
            return
        if self._pre:
            self._emit(data)
        elif data.strip():
            # Collapse the f-string indentation whitespace HTML ignores anyway.
            self._emit(re.sub(r"\s+", " ", data))


def from_html(html: str) -> str:
    conv = _Converter()
    conv.feed(html)
    conv.close()
    md = "".join(conv.out)
    md = re.sub(r"[ \t]+\n", "\n", md)      # trailing spaces
    md = re.sub(r"\n{3,}", "\n\n", md)      # collapse blank-line runs
    return md.strip() + "\n"


def token_estimate(md: str) -> int:
    """Rough token count (~4 chars/token) for the x-markdown-tokens header."""
    return max(1, len(md) // 4)
