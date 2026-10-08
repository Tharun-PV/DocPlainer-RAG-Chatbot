from docplainer.ingestion import tables


def test_clean_rows_drops_empty_rows_and_columns_and_pads():
    grid = tables.clean_rows([["A", None, "B"], ["", "", ""], ["1", "", "2", "extra"]])
    assert grid == [["A", "B", ""], ["1", "2", "extra"]]


def test_markdown_roundtrip_and_pipe_escaping():
    grid = tables.clean_rows([["Item", "Cost"], ["a|b", "5"]])
    md = tables.to_markdown(grid)
    assert md.splitlines()[1].replace(" ", "") == "|---|---|"
    assert tables.parse_markdown_table(md) == [["Item", "Cost"], ["a/b", "5"]]


def test_is_meaningful_rejects_layout_artefacts():
    assert not tables.is_meaningful([["only one row", "x"]])
    assert not tables.is_meaningful([["a"], ["b"]])
    assert not tables.is_meaningful([["a", "", ""], ["", "", ""], ["", "", "b"]])
    assert tables.is_meaningful([["h1", "h2"], ["1", "2"]])


def test_describe_mentions_columns_and_row_labels():
    grid = [["Region", "FY2024"], ["Europe", "1"], ["Asia", "2"]]
    text = tables.describe(grid, "Table 1: Revenue")
    assert "Revenue" in text and "Region" in text and "Europe" in text and "2 data rows" in text
