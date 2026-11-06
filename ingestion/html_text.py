import re
import html as _html
import unicodedata

_SKIP = re.compile(r"(?is)<(script|style|head)\b.*?</\1>")
_BLOCK = re.compile(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6]|table|ul|ol|section|article)>")
_CELL = re.compile(r"(?i)</t[dh]>")
_TAG = re.compile(r"<[^>]+>")


def html_to_text(raw: str) -> str:
    s = _SKIP.sub(" ", raw)
    s = _BLOCK.sub("\n", s)            # chỉ ngắt dòng ở thẻ khối
    s = _CELL.sub(" ", s)
    s = _TAG.sub("", s)                # thẻ inline bị bỏ, không ngắt dòng
    s = _html.unescape(s).replace("\xa0", " ")
    s = unicodedata.normalize("NFC", s)
    lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in s.splitlines())
    return "\n".join(ln for ln in lines if ln)