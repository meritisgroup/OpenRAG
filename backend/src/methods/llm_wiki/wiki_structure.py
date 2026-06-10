import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class PageType(Enum):
    ENTITY = "entities"
    CONCEPT = "concepts"
    SOURCE = "sources"
    QUERY = "queries"
    SYNTHESIS = "synthesis"


@dataclass
class WikiPage:
    title: str
    page_type: PageType
    slug: str
    frontmatter: dict = field(default_factory=dict)
    content: str = ""
    wikilinks: list = field(default_factory=list)


def slugify(title: str) -> str:
    text = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    )
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def parse_frontmatter(md_text: str) -> dict:
    if not md_text.startswith("---"):
        return {}
    end = md_text.find("---", 3)
    if end == -1:
        return {}
    yaml_str = md_text[3:end].strip()
    try:
        return yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        return {}


def render_frontmatter(meta: dict) -> str:
    if not meta:
        return ""
    yaml_str = yaml.dump(
        meta, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
    return f"---\n{yaml_str}---\n"


def parse_wikilinks(text: str) -> list:
    pattern = r"\[\[([^\]]+)\]\]"
    matches = re.findall(pattern, text)
    links = []
    for m in matches:
        if "|" in m:
            links.append(m.split("|")[0].strip())
        else:
            links.append(m.strip())
    return links


def parse_page(md_text: str) -> tuple:
    fm = parse_frontmatter(md_text)
    if md_text.startswith("---"):
        end = md_text.find("---", 3)
        if end != -1:
            content = md_text[end + 3 :].strip()
        else:
            content = md_text
    else:
        content = md_text
    return fm, content


def build_page_path(base_dir: Path, page_type: PageType, slug: str) -> Path:
    return base_dir / page_type.value / f"{slug}.md"
