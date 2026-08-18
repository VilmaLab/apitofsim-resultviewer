import asyncio
import csv
import io
import json
import os
import tempfile
from pathlib import Path

import duckdb
import pytest
from starlette.requests import Request


TEST_DIRECTORY = tempfile.TemporaryDirectory(prefix="resultviewer-tests-")
DATABASE_PATH = Path(TEST_DIRECTORY.name) / "test.duckdb"
connection = duckdb.connect(str(DATABASE_PATH))
connection.execute(
    "create table cluster_report (group_id integer, id integer, name varchar)"
)
connection.executemany(
    "insert into cluster_report values (?, ?, ?)",
    [(row % 3, row, f"row-{row:03}") for row in range(205)],
)
connection.execute("create table pathway_report (id integer)")
connection.close()

os.environ["DATABASE"] = str(DATABASE_PATH)
os.environ.setdefault("MPLCONFIGDIR", str(Path(TEST_DIRECTORY.name) / "matplotlib"))

import main  # noqa: E402


def request(query_string):
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": query_string.encode(),
        }
    )


def call(handler, query_string):
    return asyncio.run(handler(request(query_string)))


@pytest.fixture(scope="module", autouse=True)
def close_database():
    yield
    main.db.close()
    TEST_DIRECTORY.cleanup()


class TestReportData:
    def test_pages_include_total_metadata(self):
        first = call(
            main.report_data,
            "report=cluster-report&page=1&size=100",
        )
        last = call(
            main.report_data,
            "report=cluster-report&page=3&size=100",
        )
        out_of_range = call(
            main.report_data,
            "report=cluster-report&page=4&size=100",
        )

        first_payload = json.loads(first.body)
        last_payload = json.loads(last.body)
        out_of_range_payload = json.loads(out_of_range.body)
        assert first.status_code == 200
        assert first_payload["last_page"] == 3
        assert first_payload["last_row"] == 205
        assert len(first_payload["data"]) == 100
        assert len(last_payload["data"]) == 5
        assert out_of_range_payload["data"] == []

    def test_empty_report_still_has_one_page(self):
        response = call(
            main.report_data,
            "report=pathway-report&page=1&size=100",
        )
        assert json.loads(response.body) == {
            "last_page": 1,
            "last_row": 0,
            "data": [],
        }

    def test_remote_multi_column_sort_applies_before_pagination(self):
        response = call(
            main.report_data,
            "report=cluster-report&page=1&size=100"
            "&sort%5B0%5D%5Bfield%5D=group_id"
            "&sort%5B0%5D%5Bdir%5D=asc"
            "&sort%5B1%5D%5Bfield%5D=id"
            "&sort%5B1%5D%5Bdir%5D=desc",
        )
        rows = json.loads(response.body)["data"]
        assert [row["id"] for row in rows[:3]] == [204, 201, 198]

    @pytest.mark.parametrize(
        "query",
        [
            "report=unknown&page=1&size=100",
            "report=cluster-report&page=0&size=100",
            "report=cluster-report&page=1&size=50",
            "report=cluster-report&page=1&size=100"
            "&sort%5B0%5D%5Bfield%5D=missing"
            "&sort%5B0%5D%5Bdir%5D=asc",
            "report=cluster-report&page=1&size=100"
            "&sort%5B0%5D%5Bfield%5D=id"
            "&sort%5B0%5D%5Bdir%5D=sideways",
        ],
    )
    def test_invalid_parameters_are_rejected(self, query):
        assert call(main.report_data, query).status_code == 400


class TestReportDownload:
    def test_csv_contains_the_unsorted_complete_report_without_an_index(self):
        response = call(
            main.report_download,
            "report=cluster-report"
            "&sort%5B0%5D%5Bfield%5D=id"
            "&sort%5B0%5D%5Bdir%5D=desc",
        )
        rows = list(csv.DictReader(io.StringIO(response.body.decode())))

        assert response.status_code == 200
        assert response.media_type == "text/csv"
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="cluster-report.csv"'
        )
        assert list(rows[0]) == ["group_id", "id", "name"]
        assert len(rows) == 205
        assert [int(row["id"]) for row in rows[:3]] == [0, 1, 2]
