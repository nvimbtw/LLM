import pytest
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import store
import main


@pytest.fixture
def db(monkeypatch):
    engine = create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
    store.Base.metadata.create_all(engine)
    monkeypatch.setattr(store,'_sessions',sessionmaker(engine,expire_on_commit=False))
    yield
    engine.dispose()


@pytest.fixture
async def client(db,monkeypatch,tmp_path):
    monkeypatch.delenv('BENCH_ENGINE_SERVICE_SECRET',raising=False)
    corpus = tmp_path/'corpus'
    corpus.mkdir()
    (corpus/'english.txt').write_text('Reference prose. '*100)
    monkeypatch.setenv('BENCH_CORPUS_DIR',str(corpus))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://bench') as client:
        yield client
