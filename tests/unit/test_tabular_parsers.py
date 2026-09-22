"""Unit tests for CSVParser, JSONParser, and ParserFactory registry."""

import pytest

from app.core.exceptions import ParsingError
from app.rag.ingestion.parsers.csv_parser import CSVParser
from app.rag.ingestion.parsers.factory import ParserFactory
from app.rag.ingestion.parsers.json_parser import JSONParser


@pytest.mark.asyncio
class TestCSVParser:
    """Validate CSV parsing, header detection, row formatting, and metadata."""

    @pytest.fixture
    def parser(self) -> CSVParser:
        return CSVParser()

    async def test_parse_valid_csv(self, parser: CSVParser) -> None:
        csv_data = b"id,name,role,department\n1,Alice,Engineer,AI\n2,Bob,Architect,Backend\n"
        doc = await parser.parse(csv_data, "team_roster.csv")

        assert doc.title == "Team Roster"
        assert "Total Records: 2" in doc.content
        assert "Columns: id, name, role, department" in doc.content
        assert "Record 1: id: 1 | name: Alice | role: Engineer | department: AI" in doc.content
        assert "Record 2: id: 2 | name: Bob | role: Architect | department: Backend" in doc.content
        assert doc.metadata["row_count"] == 2
        assert doc.metadata["column_count"] == 4
        assert doc.metadata["columns"] == ["id", "name", "role", "department"]
        assert len(doc.sections) == 1

    async def test_parse_empty_csv(self, parser: CSVParser) -> None:
        doc = await parser.parse(b"", "empty.csv")
        assert doc.content == ""
        assert doc.metadata["row_count"] == 0
        assert len(doc.sections) == 0

    async def test_parse_utf8_bom_csv(self, parser: CSVParser) -> None:
        csv_data = b"\xef\xbb\xbfmodel,latency_ms,cost\ngpt-4o,450,0.005\n"
        doc = await parser.parse(csv_data, "benchmarks.csv")
        assert doc.metadata["columns"] == ["model", "latency_ms", "cost"]
        assert "Record 1: model: gpt-4o | latency_ms: 450 | cost: 0.005" in doc.content


@pytest.mark.asyncio
class TestJSONParser:
    """Validate JSON parsing, array records, nested flattening, and error handling."""

    @pytest.fixture
    def parser(self) -> JSONParser:
        return JSONParser()

    async def test_parse_json_array(self, parser: JSONParser) -> None:
        json_data = b'[{"service": "retrieval", "port": 8001}, {"service": "rerank", "port": 8002}]'
        doc = await parser.parse(json_data, "services_config.json")

        assert doc.title == "Services Config"
        assert "Total Items: 2" in doc.content
        assert "Item 1: service: retrieval | port: 8001" in doc.content
        assert "Item 2: service: rerank | port: 8002" in doc.content
        assert doc.metadata["root_type"] == "array"
        assert doc.metadata["item_count"] == 2

    async def test_parse_nested_json_object(self, parser: JSONParser) -> None:
        json_data = (
            b'{"database": {"host": "localhost", "port": 5432, "auth": {"user": "postgres"}}}'
        )
        doc = await parser.parse(json_data, "database.json")

        assert "database.host: localhost" in doc.content
        assert "database.port: 5432" in doc.content
        assert "database.auth.user: postgres" in doc.content
        assert doc.metadata["root_type"] == "object"
        assert doc.metadata["item_count"] == 3

    async def test_parse_malformed_json_raises_parsing_error(self, parser: JSONParser) -> None:
        json_data = b'{"unclosed": "brace"'
        with pytest.raises(ParsingError) as exc_info:
            await parser.parse(json_data, "corrupted.json")
        assert "Invalid JSON syntax" in str(exc_info.value)


class TestParserFactoryRegistration:
    """Verify CSV and JSON parsers are registered by default in ParserFactory."""

    def test_factory_resolves_csv_and_json(self) -> None:
        factory = ParserFactory()
        csv_parser = factory.get_parser("dataset.csv")
        json_parser = factory.get_parser("config.json")

        assert isinstance(csv_parser, CSVParser)
        assert isinstance(json_parser, JSONParser)
