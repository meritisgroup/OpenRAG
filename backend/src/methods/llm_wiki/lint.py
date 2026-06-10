import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .wiki_manager import WikiManager
from .wiki_structure import PageType, slugify, parse_wikilinks
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class LintIssue:
    category: str
    page: str
    description: str
    fixable: bool = False


@dataclass
class LintReport:
    orphans: list = field(default_factory=list)
    broken_links: list = field(default_factory=list)
    missing_pages: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    stale_pages: list = field(default_factory=list)
    health_score: int = 100
    suggestions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "health_score": self.health_score,
            "orphans": [vars(o) if hasattr(o, "__dict__") else o for o in self.orphans],
            "broken_links": [
                vars(o) if hasattr(o, "__dict__") else o for o in self.broken_links
            ],
            "missing_pages": [
                vars(o) if hasattr(o, "__dict__") else o for o in self.missing_pages
            ],
            "contradictions": [
                vars(o) if hasattr(o, "__dict__") else o for o in self.contradictions
            ],
            "stale_pages": [
                vars(o) if hasattr(o, "__dict__") else o for o in self.stale_pages
            ],
            "suggestions": self.suggestions,
        }


class WikiLinter:
    def __init__(
        self, wiki_manager: WikiManager, agent, llm_model: str, language: str = "EN"
    ):
        self.wiki_manager = wiki_manager
        self.agent = agent
        self.llm_model = llm_model
        self.language = language
        self.prompts = PROMPTS[self.language]

    def lint(self, fix: bool = False) -> LintReport:
        report = LintReport()

        all_pages = self.wiki_manager.list_pages()
        if not all_pages:
            report.health_score = 0
            report.suggestions.append(
                "Wiki is empty. Add documents and run indexation."
            )
            return report

        slug_map = {}
        for page in all_pages:
            slug_map[f"{page.page_type.value}/{page.slug}"] = page
            slug_map[page.title] = page

        self._check_orphans(all_pages, slug_map, report)
        self._check_broken_links(all_pages, slug_map, report)
        self._check_missing_pages(all_pages, slug_map, report)
        self._check_staleness(all_pages, report)
        self._llm_deep_check(all_pages, report)

        issue_count = (
            len(report.orphans)
            + len(report.broken_links)
            + len(report.missing_pages)
            + len(report.contradictions)
            + len(report.stale_pages)
        )
        total_pages = max(len(all_pages), 1)
        report.health_score = max(
            0, 100 - (issue_count * 100 // (total_pages * 5 + 10))
        )

        if fix:
            self._auto_fix(report, slug_map)

        return report

    def _check_orphans(
        self, all_pages: list, slug_map: dict, report: LintReport
    ) -> None:
        inbound = {}
        for page in all_pages:
            for link in page.wikilinks:
                inbound[link] = inbound.get(link, 0) + 1

        for page in all_pages:
            if page.page_type.value == "source":
                continue
            page_ref = page.title
            if inbound.get(page_ref, 0) == 0:
                report.orphans.append(
                    LintIssue(
                        category="orphan",
                        page=f"{page.page_type.value}/{page.slug}",
                        description=f'Page "{page.title}" has no inbound links',
                        fixable=True,
                    )
                )

    def _check_broken_links(
        self, all_pages: list, slug_map: dict, report: LintReport
    ) -> None:
        all_titles = set()
        all_slugs = set()
        for page in all_pages:
            all_titles.add(page.title)
            all_slugs.add(page.slug)
            all_slugs.add(f"{page.page_type.value}/{page.slug}")

        for page in all_pages:
            for link in page.wikilinks:
                if link in all_titles or link in all_slugs:
                    continue
                if any(link.lower() == t.lower() for t in all_titles):
                    continue
                if any(link.lower() == s.lower() for s in all_slugs):
                    continue
                report.broken_links.append(
                    LintIssue(
                        category="broken_link",
                        page=f"{page.page_type.value}/{page.slug}",
                        description=f'Link "[[{link}]]" points to non-existent page',
                        fixable=True,
                    )
                )

    def _check_missing_pages(
        self, all_pages: list, slug_map: dict, report: LintReport
    ) -> None:
        existing_slugs = set()
        for page in all_pages:
            existing_slugs.add(page.slug)
            existing_slugs.add(slugify(page.title))

        mentioned = {}
        for page in all_pages:
            for link in page.wikilinks:
                link_slug = slugify(link)
                if link_slug not in existing_slugs:
                    mentioned.setdefault(link, []).append(
                        f"{page.page_type.value}/{page.slug}"
                    )

        for link, sources in mentioned.items():
            already_reported = any(
                bl.description.endswith(link)
                for bl in report.broken_links
                if isinstance(bl, LintIssue)
            )
            if not already_reported:
                report.missing_pages.append(
                    LintIssue(
                        category="missing_page",
                        page=link,
                        description=f'Page "[[{link}]]" is referenced but does not exist (from: {", ".join(sources[:3])})',
                        fixable=True,
                    )
                )

    def _check_staleness(self, all_pages: list, report: LintReport) -> None:
        today = date.today()
        for page in all_pages:
            updated_str = page.frontmatter.get("updated", "")
            if not updated_str:
                continue
            try:
                updated_date = date.fromisoformat(updated_str)
                days_old = (today - updated_date).days
                if days_old > 90:
                    report.stale_pages.append(
                        LintIssue(
                            category="stale",
                            page=f"{page.page_type.value}/{page.slug}",
                            description=f'Page "{page.title}" has not been updated in {days_old} days',
                            fixable=False,
                        )
                    )
            except ValueError:
                pass

    def _llm_deep_check(self, all_pages: list, report: LintReport) -> None:
        index_text = self.wiki_manager.read_index()
        schema_text = self.wiki_manager.read_schema()
        link_map = self.wiki_manager.get_all_wikilinks()

        pages_content_parts = []
        for page in all_pages[:20]:
            pages_content_parts.append(
                f"## {page.title} ({page.page_type.value})\n{page.content[:300]}\n"
            )
        pages_content = "\n".join(pages_content_parts)

        prompt_template = self.prompts["lint_report"]["QUERY_TEMPLATE"]
        system_prompt = self.prompts["lint_report"]["SYSTEM_PROMPT"]

        prompt = prompt_template.format(
            schema=schema_text,
            index=index_text,
            pages_content=pages_content,
            link_map=json.dumps(link_map, ensure_ascii=False, indent=2),
        )

        try:
            result = self.agent.predict(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.llm_model,
            )
            llm_report = self._parse_json_response(result["texts"], {})
        except Exception as e:
            logger.warning(f"LLM deep lint check failed: {e}")
            return

        if not llm_report:
            return

        for c in llm_report.get("contradictions", []):
            report.contradictions.append(
                LintIssue(
                    category="contradiction",
                    page=f"{c.get('page1', '')} / {c.get('page2', '')}",
                    description=c.get("description", ""),
                    fixable=False,
                )
            )

        for s in llm_report.get("suggestions", []):
            report.suggestions.append(s)

    def _parse_json_response(self, text: str, default):
        try:
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]
            elif "[" in text:
                start = text.index("[")
                end = text.rindex("]") + 1
                json_str = text[start:end]
            else:
                return default
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return default

    def _auto_fix(self, report: LintReport, slug_map: dict) -> None:
        all_pages = self.wiki_manager.list_pages()

        for issue in report.broken_links:
            if not isinstance(issue, LintIssue) or not issue.fixable:
                continue
            link_target = issue.description.split('"')[1] if '"' in issue.description else ""
            if not link_target:
                continue
            slug = slugify(link_target)
            stub_frontmatter = {
                "title": link_target,
                "type": "entity",
                "sources": [],
                "tags": ["stub", "auto-generated"],
            }
            stub_content = f"# {link_target}\n\n*This is a stub page. Content needs to be added.*\n"
            self.wiki_manager.write_page(PageType.ENTITY, slug, stub_content, stub_frontmatter)
            report.suggestions.append(
                f"Created stub page for [[{link_target}]]"
            )

        for issue in report.orphans:
            if not isinstance(issue, LintIssue) or not issue.fixable:
                continue
            page_ref = issue.page
            if page_ref not in slug_map:
                continue
            orphan_page = slug_map[page_ref]
            for candidate in all_pages:
                if candidate.page_type == orphan_page.page_type:
                    continue
                if candidate.slug == orphan_page.slug:
                    continue
                candidate.content += f"\n\nSee also: [[{orphan_page.title}]]\n"
                candidate.wikilinks = parse_wikilinks(candidate.content)
                fm = candidate.frontmatter
                self.wiki_manager.write_page(
                    candidate.page_type, candidate.slug, candidate.content, fm
                )
                report.suggestions.append(
                    f'Added link to [[{orphan_page.title}]] from {candidate.page_type.value}/{candidate.slug}'
                )
                break

        self.wiki_manager.update_index()
