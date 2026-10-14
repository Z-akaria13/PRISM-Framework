"""core.metadata.read_metadata: the ID column is read as text (debt P1 #25)."""
from prism.core.metadata import read_metadata


def test_the_id_column_keeps_its_text_and_other_columns_their_types(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("scan,age,label\n0042,71.5,1\n1e3,64,0\n007,80,1\n")

    metadata = read_metadata(path, "scan")

    assert list(metadata["scan"]) == ["0042", "1e3", "007"]
    assert metadata["age"].dtype.kind == "f" and metadata["label"].dtype.kind == "i"
    assert list(read_metadata(path)["scan"]) == [42.0, 1000.0, 7.0]  # what pandas makes of them by default


def test_an_id_column_that_is_not_there_is_left_to_the_caller(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("scan,label\n0042,1\n")

    assert list(read_metadata(path, "subject_id")["scan"]) == [42]  # the run then says the column is missing
