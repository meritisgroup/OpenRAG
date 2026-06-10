import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

from .wiki_structure import (
    PageType,
    WikiPage,
    build_page_path,
    parse_frontmatter,
    parse_page,
    parse_wikilinks,
    render_frontmatter,
    slugify,
)

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA = """# Wiki Schema

## Conventions
- Pages are written in markdown with YAML frontmatter
- Use wikilinks [[Page Name]] to reference other pages
- Use standard links [Name](path.md) for compatibility

## Page Types
- **entity**: People, organizations, products, places
- **concept**: Theories, methods, techniques, ideas
- **source**: Summaries of ingested documents
- **query**: Saved interesting Q&A pairs
- **synthesis**: Cross-source analyses

## Naming Rules
- Entity pages: created for named things mentioned in sources
- Concept pages: created for abstract ideas or methods
- A source document should always have a corresponding source page

## Tone
- Factual, concise, encyclopedic
- Use cross-references liberally
- Each page should be self-contained but linked to related pages
"""

_DEFAULT_SCHEMA_FR = """# Schema du Wiki

## Conventions
- Les pages sont ecrites en markdown avec frontmatter YAML
- Utiliser les wikilinks [[Nom de Page]] pour referencer d'autres pages
- Utiliser les liens standards [Nom](chemin.md) pour la compatibilite

## Types de Pages
- **entity**: Personnes, organisations, produits, lieux
- **concept**: Theories, methodes, techniques, idees
- **source**: Resumes des documents ingeres
- **query**: Paires Q&R interesantes sauvegardees
- **synthesis**: Analyses cross-sources

## Regles de Nommage
- Pages entity: creees pour les choses nommees mentionnees dans les sources
- Pages concept: creees pour les idees abstraites ou methodes
- Un document source devrait toujours avoir une page source correspondante

## Ton
- Factuel, concis, encyclopedique
- Utiliser les cross-references liberallement
- Chaque page doit etre autonome mais liee aux pages connexes
"""


class WikiManager:
    def __init__(self, wiki_path: Path, language: str = "EN"):
        self.wiki_path = Path(wiki_path)
        self.language = language

    def initialize(self) -> None:
        for pt in PageType:
            (self.wiki_path / pt.value).mkdir(parents=True, exist_ok=True)

        if not (self.wiki_path / "index.md").exists():
            self._write_file("index.md", "# Wiki Index\n\nNo pages yet.\n")

        if not (self.wiki_path / "log.md").exists():
            self._write_file("log.md", "# Wiki Log\n\n")

        if not (self.wiki_path / "schema.md").exists():
            schema = _DEFAULT_SCHEMA_FR if self.language == "FR" else _DEFAULT_SCHEMA
            self._write_file("schema.md", schema)

        if not (self.wiki_path / "overview.md").exists():
            self._write_file("overview.md", "# Wiki Overview\n\nNo content yet.\n")

    def _read_file(self, relative_path: str) -> str:
        path = self.wiki_path / relative_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_file(self, relative_path: str, content: str) -> None:
        path = self.wiki_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def read_page(self, page_type: PageType, slug: str) -> Optional[WikiPage]:
        path = build_page_path(self.wiki_path, page_type, slug)
        if not path.exists():
            return None
        md_text = path.read_text(encoding="utf-8")
        fm, content = parse_page(md_text)
        return WikiPage(
            title=fm.get("title", slug),
            page_type=page_type,
            slug=slug,
            frontmatter=fm,
            content=content,
            wikilinks=parse_wikilinks(content),
        )

    def write_page(
        self, page_type: PageType, slug: str, content: str, frontmatter: dict
    ) -> None:
        fm_text = render_frontmatter(frontmatter)
        full_text = f"{fm_text}\n{content}\n" if fm_text else f"{content}\n"
        path = build_page_path(self.wiki_path, page_type, slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_text, encoding="utf-8")

    def delete_page(self, page_type: PageType, slug: str) -> None:
        path = build_page_path(self.wiki_path, page_type, slug)
        if path.exists():
            path.unlink()

    def list_pages(self, page_type: Optional[PageType] = None) -> list:
        pages = []
        if page_type:
            types_to_scan = [page_type]
        else:
            types_to_scan = list(PageType)

        for pt in types_to_scan:
            pt_dir = self.wiki_path / pt.value
            if not pt_dir.exists():
                continue
            for md_file in pt_dir.glob("*.md"):
                page_slug = md_file.stem
                page = self.read_page(pt, page_slug)
                if page:
                    pages.append(page)
        return pages

    def search_pages(self, query_text: str) -> list:
        query_lower = query_text.lower()
        results = []
        for page in self.list_pages():
            if query_lower in page.title.lower() or query_lower in page.content.lower():
                results.append(page)
        return results

    def page_exists(self, page_type: PageType, slug: str) -> bool:
        path = build_page_path(self.wiki_path, page_type, slug)
        return path.exists()

    def read_index(self) -> str:
        return self._read_file("index.md")

    def read_schema(self) -> str:
        return self._read_file("schema.md")

    def read_overview(self) -> str:
        return self._read_file("overview.md")

    def update_index(self) -> None:
        pages = self.list_pages()
        lines = ["# Wiki Index\n"]
        for pt in PageType:
            pt_pages = [p for p in pages if p.page_type == pt]
            if pt_pages:
                lines.append(f"\n## {pt.value.capitalize()}\n")
                for p in sorted(pt_pages, key=lambda x: x.title):
                    rel_path = f"{pt.value}/{p.slug}.md"
                    summary = p.frontmatter.get("summary", "")
                    source_count = len(p.frontmatter.get("sources", []))
                    updated = p.frontmatter.get("updated", "")
                    meta_parts = []
                    if source_count:
                        meta_parts.append(f"{source_count} source{'s' if source_count > 1 else ''}")
                    if updated:
                        meta_parts.append(updated)
                    meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
                    if summary:
                        lines.append(f"- [[{p.title}]] — [{p.title}]({rel_path}){meta}: {summary}\n")
                    else:
                        lines.append(f"- [[{p.title}]] — [{p.title}]({rel_path}){meta}\n")

        self._write_file("index.md", "".join(lines))

    def update_overview(self, overview_text: str) -> None:
        self._write_file("overview.md", overview_text)

    def append_log(self, operation: str, details: str) -> None:
        today = date.today().isoformat()
        entry = f"\n## [{today}] {operation} | {details}\n"
        log_path = self.wiki_path / "log.md"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def get_all_wikilinks(self) -> dict:
        link_map = {}
        for page in self.list_pages():
            for link in page.wikilinks:
                if link not in link_map:
                    link_map[link] = []
                link_map[link].append(f"{page.page_type.value}/{page.slug}")
        return link_map

    def get_all_slugs(self) -> dict:
        slug_map = {}
        for page in self.list_pages():
            slug_map[page.title] = (page.page_type, page.slug)
        return slug_map
