from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TocChapter:
    title: str
    subsections: list[str] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentSection:
    section_id: str
    chapter: str
    subsection: str
    heading: str
    page_start: int
    page_end: int
    order: int
    text: str
    headings_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedDocument:
    title: str
    source_path: str
    total_pages: int
    chapters: list[TocChapter]
    sections: list[DocumentSection]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source_path": self.source_path,
            "total_pages": self.total_pages,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "sections": [section.to_dict() for section in self.sections],
        }

