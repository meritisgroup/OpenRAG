from sqlalchemy import String, Integer, JSON, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


from ..utils.open_doc import Opener
from ..utils.splitter import TextSplitter
from ..utils.base_classes import Splitter


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True)
    document: Mapped[str] = mapped_column(String(), primary_key=True)
    text: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"Chunk(id={self.id!r}, doc={self.document!r}, text={self.text!r})"


class ChunkPath(Base):
    __tablename__ = "chunks_image"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True)
    doc_id: Mapped[int] = mapped_column(Integer(), primary_key=True)
    document: Mapped[str] = mapped_column(String(), primary_key=True)
    path: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"Chunk(id={self.id!r}, doc={self.document!r}, path={self.path!r}, , doc_id={self.doc_id!r})"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String())
    embedding_tokens: Mapped[int] = mapped_column(Integer())
    input_tokens: Mapped[int] = mapped_column(Integer())
    output_tokens: Mapped[int] = mapped_column(Integer())

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, name={self.name!r}, tokens={self.tokens!r}, input tokens = {self.input_tokens!r}, output tokens = {self.output_tokens!r})"


class DocumentPath(Base):
    __tablename__ = "documents_image"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String())
    path: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"DocumentImage(id={self.id!r}, name={self.name!r}, path={self.path!r})"


class DocumentText:
    """ Class used to open, chunk and store documents"""
    def __init__(self, path: str, splitter: Splitter = TextSplitter()):

        self.name_with_extension = path.split("/")[-1]
        try:
            self.content = Opener(save=False).open_doc(path)

        except Exception as e:
            self.content = ""
            print(f'Error "{e}" while trying to open doc {self.name_with_extension}')

        self.name = ".".join(self.name_with_extension.split(".")[:-1])
        self.extension = "." + self.name_with_extension.split(".")[-1]
        self.text_splitter = splitter
        self.embedding_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def chunks(self, chunk_size: int = 500, chunk_overlap: bool = True) -> list[Chunk]:
        """
        Split the text according to parameters and return associated chunks
        """
        chunks = self.text_splitter.split_text(
            text=self.content, chunk_size=chunk_size, overlap=chunk_overlap
        )

        results = []

        for k, chunk in enumerate(chunks):
            results.append(
                Chunk(text=chunk, document=self.name_with_extension, id=k + 1)
            )

        return results

    def convert_in_base(self) -> Document:
        return Document(
            name=self.name_with_extension,
            embedding_tokens=self.embedding_tokens,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String())
    kind: Mapped[str] = mapped_column(String())
    description: Mapped[str] = mapped_column(String())
    chunk_id: Mapped[int] = mapped_column(Integer())
    doc_name: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"Entity(id={self.id!r}, name={self.name!r}, kind={self.kind!r}, description={self.description!r}, chunk_id={self.chunk_id!r}, doc_name={self.doc_name!r})"


class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String())
    target: Mapped[str] = mapped_column(String())
    description: Mapped[str] = mapped_column(String())
    weight: Mapped[float] = mapped_column(Float())
    keywords: Mapped[str] = mapped_column(String())
    chunk_id: Mapped[int] = mapped_column(Integer())
    doc_name: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"Relation(id={self.id!r}, entity_src={self.source!r}, entity_cbl={self.target!r}, relation_description={self.description!r}, chunk={self.chunk_id!r}, doc={self.doc_name!r})"


class Community(Base):
    __tablename__ = "communities"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String())
    description: Mapped[str] = mapped_column(String())
    entities_ids: Mapped[JSON] = mapped_column(JSON())
    hierarchical_level: Mapped[int] = mapped_column(Integer())
    # doc_name: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"Community(id={self.id!r}, title={self.title!r}, description={self.description!r}, entities_ids={str(self.entities_ids)!r})"


class Tokens(Base):
    __tablename__ = "tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String())
    embedding_tokens: Mapped[int] = mapped_column(Integer())
    input_tokens: Mapped[int] = mapped_column(Integer())
    output_tokens: Mapped[int] = mapped_column(Integer())

    def __repr__(self) -> str:
        return f"Tokens(id={self.id!r}, title={self.title!r}, embedding_tokens={self.embedding_tokens!r}, intput_tokens={str(self.input_tokens)!r}, output_tokens={str(self.output_tokens)!r})"


class Question(Base):
    __tablename__ = "cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String())
    chunk_id: Mapped[str] = mapped_column(String())
    doc_name: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"Question(id={self.id!r}, text={self.text!r}, chunk={self.chunk_id!r}, doc={self.doc_name!r})"


class MergeEntityOverall(Base):
    __tablename__ = "merge_entities_overall"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String())
    kind: Mapped[str] = mapped_column(String())
    description: Mapped[str] = mapped_column(String())
    chunk_ids: Mapped[JSON] = mapped_column(JSON())
    doc_names: Mapped[JSON] = mapped_column(JSON())
    degree: Mapped[int] = mapped_column(Integer())

    def __repr__(self) -> str:
        return f"Entity(id={self.id!r}, name={self.name!r}, kind={self.kind!r}, description={self.description!r}, chunk_ids={str(self.chunk_ids)!r}, doc_names={str(self.doc_names)!r}, degree={str(self.degree)!r})"


class MergeEntityDocument(Base):
    __tablename__ = "merge_entities_document"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String())
    kind: Mapped[str] = mapped_column(String())
    description: Mapped[str] = mapped_column(String())
    chunk_ids: Mapped[JSON] = mapped_column(JSON())
    doc_name: Mapped[str] = mapped_column(String())
    degree: Mapped[int] = mapped_column(Integer())

    def __repr__(self) -> str:
        return f"Entity(id={self.id!r}, name={self.name!r}, kind={self.kind!r}, description={self.description!r}, chunk_ids={str(self.chunk_ids)!r}, doc_names={self.doc_name!r}, degree={str(self.degree)!r})"


class CommunityRelation(Base):
    __tablename__ = "communities_relations"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String())
    target: Mapped[str] = mapped_column(String())
    description: Mapped[str] = mapped_column(String())

    def __repr__(self) -> str:
        return f"CommunityRelation(id={self.id!r}, entity_src={self.source!r}, entity_cbl={self.target!r}, relation_description={self.description!r})"


class CommunityEntity(Base):
    __tablename__ = "communities_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String())
    kind: Mapped[str] = mapped_column(String())
    description: Mapped[str] = mapped_column(String())
    entities_ids: Mapped[JSON] = mapped_column(JSON())
    doc_name: Mapped[str] = mapped_column(String())
    degree: Mapped[int] = mapped_column(Integer())

    def __repr__(self) -> str:
        return f"CommunityEntity(id={self.id!r}, name={self.name!r}, kind={self.kind!r}, description={self.description!r}, chunk_ids={str(self.entities_ids)!r}, doc_names={self.doc_name!r}, degree={str(self.degree)!r})"
