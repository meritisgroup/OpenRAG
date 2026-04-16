import secrets
import string
from dataclasses import dataclass, field
from typing import Optional

_ALPHABET = string.ascii_letters + string.digits


def _make_id() -> str:
    return ''.join(secrets.choice(_ALPHABET) for _ in range(15))


@dataclass
class SectionNode:
    level: int
    title: str
    text: str
    document: str
    position: int = 0
    parent_id: Optional[str] = None
    id: str = field(default_factory=_make_id)
    summary: str = ''
    children: list['SectionNode'] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'parent_id': self.parent_id or '',
            'document': self.document,
            'level': self.level,
            'title': self.title,
            'summary': self.summary,
            'text': self.text,
            'position': self.position,
        }

    def flatten(self) -> list['SectionNode']:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.flatten())
        return nodes

    @classmethod
    def from_dict(cls, data: dict) -> 'SectionNode':
        return cls(
            id=data.get('id', _make_id()),
            parent_id=data.get('parent_id'),
            document=data.get('document', ''),
            level=data.get('level', 0),
            title=data.get('title', ''),
            summary=data.get('summary', ''),
            text=data.get('text', ''),
            position=data.get('position', 0),
        )


def build_tree_from_chunks(chunks: list[str], document_name: str, max_section_size: int = 2000, max_paragraph_size: int = 800) -> list[SectionNode]:
    root_nodes = []
    current_section = None
    section_position = 0
    paragraph_position = 0

    for chunk in chunks:
        is_header = chunk.strip().startswith('#') and '\n' in chunk
        if is_header or current_section is None:
            if current_section is not None:
                root_nodes.append(current_section)
            section_position += 1
            paragraph_position = 0
            title_line = chunk.strip().split('\n')[0].lstrip('#').strip()
            current_section = SectionNode(
                level=1,
                title=title_line or f'Section {section_position}',
                text=chunk,
                document=document_name,
                position=section_position,
            )
        else:
            paragraph_position += 1
            para_node = SectionNode(
                level=2,
                title=f'{current_section.title} - Part {paragraph_position}',
                text=chunk,
                document=document_name,
                position=paragraph_position,
                parent_id=current_section.id,
            )
            current_section.children.append(para_node)

    if current_section is not None:
        root_nodes.append(current_section)

    return root_nodes


def build_document_node(document_name: str, sections: list[SectionNode]) -> SectionNode:
    combined_text = '\n\n'.join(s.text for s in sections)
    doc_node = SectionNode(
        level=0,
        title=document_name,
        text=combined_text,
        document=document_name,
        position=0,
    )
    for section in sections:
        section.parent_id = doc_node.id
        doc_node.children.append(section)
    return doc_node
