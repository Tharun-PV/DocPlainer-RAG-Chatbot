import pytest

from docplainer.config import Settings
from docplainer.session import LimitReached, RagSession


def make(**kw):
    return RagSession(Settings(embedder_backend="hashing", use_reranker=False, **kw))


def test_upload_dedup_and_removal(sample_dir):
    s = make()
    f = sample_dir / "northwind_handbook.pdf"
    _, added = s.add_files([f], count_upload=False)
    assert added > 0 and list(s.docs) == ["northwind_handbook.pdf"]
    _, again = s.add_files([f], count_upload=False)
    assert again == 0  # same content is not indexed twice
    s.remove("northwind_handbook.pdf")
    assert not s.docs and len(s.index) == 0


def test_demo_upload_limit_counts_only_real_uploads(sample_dir):
    s = make(demo_upload_limit=1)
    s.add_files([sample_dir / "northwind_handbook.pdf"], count_upload=False)  # sample: free
    s.add_files([sample_dir / "northwind_annual_report.pdf"])
    with pytest.raises(LimitReached):
        s.add_files([sample_dir / "northwind_scanned_memo.pdf"])
    s.demo_mode = False  # own key: no cap
    s.add_files([sample_dir / "northwind_scanned_memo.pdf"])


def test_question_limit_only_applies_in_demo_mode():
    s = make(demo_question_limit=2)
    s.questions_used = 2
    with pytest.raises(LimitReached):
        s.check_can_ask()
    s.demo_mode = False
    s.check_can_ask()
    assert s.questions_left is None


def test_doc_filter_none_means_everything(sample_dir):
    s = make()
    s.add_files(sorted(sample_dir.glob("*.pdf")), count_upload=False)
    names = list(s.docs)
    assert s.selected_doc_ids(names) is None and s.selected_doc_ids([]) is None
    assert s.selected_doc_ids(names[:1]) == {s.docs[names[0]]}
