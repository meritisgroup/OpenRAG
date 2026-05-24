"""
Tests for VectorBase_embeddings_faiss.

Stubs heavy transitive dependencies so the FAISS vector store
can be tested in isolation without the full backend installed.
"""
import os
import sys
import shutil
import tempfile
import types
from unittest.mock import MagicMock


# ── Stub heavy transitive imports before touching src/ ──

class _MockModule(types.ModuleType):
    def __init__(self, name='mock'):
        super().__init__(name)
    def __getattr__(self, name):
        return MagicMock()

for _mod in [
    'openai', 'mistralai', 'anthropic', 'cohere',
    'crawl4ai', 'pydub', 'pyvis', 'plotly', 'ecologits',
    'langchain_community', 'langchain_community.document_loaders',
    'sklearn', 'sklearn.metrics', 'sklearn.metrics.pairwise',
    'igraph', 'leidenalg', 'pymupdf', 'fitz',
    'pptx', 'docx', 'PIL', 'PIL.Image', 'cv2', 'reportlab',
    'elasticsearch', 'elasticsearch.helpers', 'pymilvus', 'kaleido',
    'rapidfuzz', 'rapidfuzz.fuzz',
    'unidecode', 'tiktoken', 'pydantic_settings',
    'google', 'google.generativeai',
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = _MockModule(_mod)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Now safe to import ──

import numpy as np
import pytest

from utils.vectorbase_faiss import VectorBase_embeddings_faiss
from database.rag_classes import Chunk


DIMENSION = 8


class FakeAgent:
    """Returns deterministic embeddings seeded by text content."""

    def embeddings(self, texts, model=None):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in str(t)) % (2**31)
            rng = np.random.RandomState(seed)
            vecs.append(rng.randn(DIMENSION).tolist())
        return {'embeddings': vecs, 'nb_tokens': len(texts) * 5}


def make_chunk(text, doc="doc1", pos=0):
    return Chunk(id=f"chunk_{pos}", position_in_doc=pos, document=doc, text=text)


@pytest.fixture
def storage_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def vb(storage_dir):
    return VectorBase_embeddings_faiss(
        vb_name="test_col",
        storage_path=storage_dir,
        agent=FakeAgent(),
        embedding_model="fake",
    )


# ───────────────────── Collection lifecycle ─────────────────────

class TestCreateCollection:

    def test_create_default(self, vb):
        vb.create_collection()
        assert vb.check_collection_exist("test_col")

    def test_create_named(self, vb):
        vb.create_collection(name="custom")
        assert vb.check_collection_exist("custom")
        assert not vb.check_collection_exist("nonexistent")

    def test_idempotent(self, vb, capsys):
        vb.create_collection()
        vb.create_collection()
        assert "already exists" in capsys.readouterr().out


class TestDeleteCollection:

    def test_removes_memory_and_disk(self, vb, storage_dir):
        vb.create_collection()
        vb.add_str_batch_elements([make_chunk("x")], display_message=False)
        cdir = os.path.join(storage_dir, "faiss", "test_col")
        assert os.path.exists(cdir)
        vb.delete_collection()
        assert not vb.check_collection_exist("test_col")
        assert not os.path.exists(cdir)

    def test_nonexistent(self, vb, capsys):
        vb.delete_collection(vb_name="nope")
        assert "does not exist" in capsys.readouterr().out


# ───────────────────── Add + Search ─────────────────────

class TestAddAndSearch:

    def test_batch_add_and_search(self, vb):
        vb.create_collection()
        chunks = [
            make_chunk("the cat sat on the mat", pos=0),
            make_chunk("the dog ran in the park", pos=1),
            make_chunk("machine learning is great", pos=2),
        ]
        tokens = vb.add_str_batch_elements(chunks, display_message=False)
        assert tokens > 0

        results = vb.k_search(queries=["cat mat"], k=2)
        assert len(results) == 1
        assert len(results[0]) == 2
        assert all(isinstance(c, Chunk) for c in results[0])

    def test_add_elements_delegates(self, vb):
        vb.create_collection()
        tokens = vb.add_str_elements([make_chunk("hello")], display_message=False)
        assert tokens > 0
        assert len(vb.k_search(queries=["hello"], k=1)[0]) == 1

    def test_add_empty(self, vb):
        vb.create_collection()
        assert vb.add_str_batch_elements([], display_message=False) == 0

    def test_returned_chunk_fields(self, vb):
        vb.create_collection()
        vb.add_str_batch_elements(
            [make_chunk("quantum computing", doc="physics.pdf", pos=5)],
            display_message=False,
        )
        chunk = vb.k_search(queries=["quantum"], k=1)[0][0]
        assert chunk.text == "quantum computing"
        assert chunk.document == "physics.pdf"
        assert chunk.position_in_doc == 5
        assert chunk.id == "chunk_5"

    def test_k_larger_than_index(self, vb):
        vb.create_collection()
        vb.add_str_batch_elements([make_chunk("only one")], display_message=False)
        assert len(vb.k_search(queries=["any"], k=100)[0]) == 1

    def test_search_empty_collection(self, vb):
        vb.create_collection()
        assert vb.k_search(queries=["any"], k=5) == [[]]

    def test_search_nonexistent_collection(self, vb):
        assert vb.k_search(queries=["any"], k=5, collection_name="nope") == [[]]

    def test_multiple_queries(self, vb):
        vb.create_collection()
        vb.add_str_batch_elements(
            [make_chunk("alpha beta", pos=0), make_chunk("delta epsilon", pos=1)],
            display_message=False,
        )
        results = vb.k_search(queries=["alpha", "delta"], k=2)
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2

    def test_auto_creates_collection(self, vb):
        vb.add_str_batch_elements(
            [make_chunk("auto")], display_message=False, collection_name="auto"
        )
        assert vb.check_collection_exist("auto")

    def test_string_query(self, vb):
        vb.create_collection()
        vb.add_str_batch_elements([make_chunk("test")], display_message=False)
        assert len(vb.k_search(queries="test", k=1)) == 1


# ───────────────────── Persistence ─────────────────────

class TestPersistence:

    def test_save_and_reload(self, storage_dir):
        agent = FakeAgent()
        vb1 = VectorBase_embeddings_faiss(
            vb_name="persist", storage_path=storage_dir,
            agent=agent, embedding_model="fake",
        )
        vb1.create_collection()
        vb1.add_str_batch_elements(
            [make_chunk("alpha", pos=0), make_chunk("beta", pos=1)],
            display_message=False,
        )
        cdir = os.path.join(storage_dir, "faiss", "persist")
        assert os.path.exists(os.path.join(cdir, "index.faiss"))
        assert os.path.exists(os.path.join(cdir, "metadata.pkl"))

        vb2 = VectorBase_embeddings_faiss(
            vb_name="persist", storage_path=storage_dir,
            agent=agent, embedding_model="fake",
        )
        assert len(vb2.k_search(queries=["alpha"], k=2)[0]) == 2

    def test_incremental_persists(self, storage_dir):
        agent = FakeAgent()
        vb = VectorBase_embeddings_faiss(
            vb_name="incr", storage_path=storage_dir,
            agent=agent, embedding_model="fake",
        )
        vb.create_collection()
        vb.add_str_batch_elements([make_chunk("first", pos=0)], display_message=False)
        vb.add_str_batch_elements([make_chunk("second", pos=1)], display_message=False)

        vb2 = VectorBase_embeddings_faiss(
            vb_name="incr", storage_path=storage_dir,
            agent=agent, embedding_model="fake",
        )
        assert len(vb2.k_search(queries=["any"], k=10)[0]) == 2


# ───────────────────── Misc ─────────────────────

class TestCheckElementExist:

    def test_found(self, vb):
        vb.create_collection()
        vb.add_str_batch_elements([make_chunk("findable")], display_message=False)
        assert vb.check_element_exist("findable")

    def test_not_found(self, vb):
        vb.create_collection()
        vb.add_str_batch_elements([make_chunk("other")], display_message=False)
        assert not vb.check_element_exist("missing")


class TestTokenTracking:

    def test_accumulates(self, vb):
        vb.create_collection()
        t1 = vb.add_str_batch_elements([make_chunk("a", pos=0)], display_message=False)
        t2 = vb.add_str_batch_elements([make_chunk("b", pos=1)], display_message=False)
        assert vb.get_nb_token_embeddings() == t1 + t2
