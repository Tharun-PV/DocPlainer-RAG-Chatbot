from docplainer.config import Settings
from docplainer.ingestion import chunk_document
from docplainer.schemas import Block, ParsedDocument, ParsedPage


def doc(*pages: ParsedPage) -> ParsedDocument:
    return ParsedDocument("d1", "report.pdf", list(pages))


def test_headings_build_section_path_and_split_chunks():
    d = doc(
        ParsedPage(
            1,
            [
                Block("heading", "Report", 1),
                Block("heading", "Revenue", 2),
                Block("paragraph", "Revenue grew strongly across all regions this year."),
                Block("heading", "People", 2),
                Block("paragraph", "Headcount rose in engineering and operations teams."),
            ],
        )
    )
    chunks = chunk_document(d, Settings())
    assert [c.section for c in chunks] == ["Report > Revenue", "Report > People"]
    assert all(c.embed_text.startswith("report.pdf | Report") for c in chunks)


def test_chunks_do_not_cross_pages_so_citations_are_exact():
    p = "Sentence number one is here. " * 5
    d = doc(ParsedPage(1, [Block("paragraph", p)]), ParsedPage(2, [Block("paragraph", p)]))
    assert [c.page for c in chunk_document(d, Settings())] == [1, 2]


def test_long_paragraph_is_split_with_sentence_overlap():
    sentences = [f"Fact {i} is stated here." for i in range(60)]
    cfg = Settings(chunk_max_chars=300, chunk_overlap_chars=60)
    chunks = chunk_document(doc(ParsedPage(1, [Block("paragraph", " ".join(sentences))])), cfg)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 60 + 5 for c in chunks)
    # overlap: the first sentence of chunk n+1 already appeared at the end of chunk n
    first = chunks[1].text.split(". ")[0]
    assert first in chunks[0].text


def test_small_table_is_one_chunk_with_caption_and_description():
    md = "| Region | FY24 |\n| --- | --- |\n| Europe | 141.9 |"
    d = doc(ParsedPage(1, [Block("table", md, caption="Table 1: Revenue")]))
    (c,) = chunk_document(d, Settings())
    assert c.kind == "table" and c.text.startswith("Table 1: Revenue")
    assert "Table with columns: Region, FY24" in c.embed_text and "141.9" in c.text


def test_large_table_splits_on_row_boundaries_and_repeats_header():
    rows = "\n".join(f"| item{i} | {i * 11} |" for i in range(80))
    md = f"| Name | Value |\n| --- | --- |\n{rows}"
    chunks = chunk_document(doc(ParsedPage(1, [Block("table", md)])), Settings(table_max_chars=400))
    assert len(chunks) > 2
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[0] == "| Name | Value |" and lines[1].startswith("| ---")
        assert all(line.count("|") == 3 for line in lines)  # no row cut in half
    body = [ln for c in chunks for ln in c.text.splitlines()[2:]]
    assert len(body) == 80


def test_custom_summarizer_is_used_for_embedding_text_only():
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    d = doc(ParsedPage(1, [Block("table", md)]))
    (c,) = chunk_document(d, Settings(), summarizer=lambda t: "CUSTOM SUMMARY")
    assert "CUSTOM SUMMARY" in c.embed_text and "CUSTOM SUMMARY" not in c.text
