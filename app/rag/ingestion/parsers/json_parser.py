"""Structured JSON document parser with hierarchical flattening and item formatting."""

import json
from pathlib import Path
from typing import Any

from app.core.exceptions import ParsingError
from app.rag.ingestion.parsers.base import DocumentParserProtocol, ParsedDocument


class JSONParser(DocumentParserProtocol):
    """Parses structured JSON documents into human-readable textual knowledge."""

    def supported_extensions(self) -> set[str]:
        return {".json"}

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Recursively flatten nested dictionary keys."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for idx, elem in enumerate(v):
                    if isinstance(elem, dict):
                        items.extend(self._flatten_dict(elem, f"{new_key}[{idx}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{idx}]", elem))
            else:
                items.append((new_key, v))
        return dict(items)

    def _is_openapi(self, data: Any) -> bool:
        """Check if parsed JSON represents an OpenAPI or Swagger document."""
        if not isinstance(data, dict):
            return False
        return bool(
            "openapi" in data
            or "swagger" in data
            or ("paths" in data and "info" in data)
        )

    def _parse_openapi(
        self, data: dict[str, Any], derived_title: str
    ) -> tuple[str, str, list[dict[str, str]], int]:
        """Transform OpenAPI/Swagger specifications into clean semantic Markdown entities."""
        info = data.get("info", {}) if isinstance(data.get("info"), dict) else {}
        title = info.get("title") or derived_title
        version = data.get("openapi") or data.get("swagger") or "3.0.0"
        api_ver = info.get("version", "")
        desc = info.get("description", "")

        paths = data.get("paths", {}) if isinstance(data.get("paths"), dict) else {}
        http_methods = {"get", "post", "put", "delete", "patch", "options", "head", "trace"}
        total_endpoints = 0

        endpoint_entries: list[dict[str, Any]] = []
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method, op in path_item.items():
                if method.lower() not in http_methods or not isinstance(op, dict):
                    continue
                total_endpoints += 1
                method_upper = method.upper()
                summary = op.get("summary", "")
                op_desc = op.get("description", "")
                tags = op.get("tags", [])
                op_id = op.get("operationId", "")
                params = op.get("parameters", [])
                responses = op.get("responses", {})
                endpoint_entries.append({
                    "method": method_upper,
                    "path": path,
                    "summary": summary,
                    "description": op_desc,
                    "tags": tags,
                    "operation_id": op_id,
                    "parameters": params,
                    "responses": responses,
                })

        header_lines: list[str] = [
            f"# {title} (OpenAPI Specification)",
            f"- Specification: OpenAPI {version}",
        ]
        if api_ver:
            header_lines.append(f"- API Version: {api_ver}")
        if desc:
            header_lines.append(f"- Description: {desc}")
        header_lines.append("")
        header_lines.append("## Available Endpoints")
        header_lines.append("")
        for ep in endpoint_entries:
            s_text = ep["summary"] or ep["description"] or "API operation"
            header_lines.append(f"- {ep['method']} {ep['path']} — {s_text}")
        header_lines.append("")

        lines: list[str] = list(header_lines)
        sections: list[dict[str, str]] = [
            {
                "title": f"{title} Overview",
                "content": "\n".join(header_lines),
            }
        ]

        for ep in endpoint_entries:
            method_upper = ep["method"]
            path = ep["path"]
            summary = ep["summary"]
            op_desc = ep["description"]
            tags = ep["tags"]
            op_id = ep["operation_id"]
            params = ep["parameters"]
            responses = ep["responses"]

            ep_lines: list[str] = [f"### Endpoint: {method_upper} {path}"]
            if tags:
                tag_str = ", ".join(str(t) for t in tags)
                ep_lines.append(f"- Tags: {tag_str}")
            if summary:
                ep_lines.append(f"- Summary: {summary}")
            if op_desc and op_desc != summary:
                ep_lines.append(f"- Description: {op_desc}")
            if op_id:
                ep_lines.append(f"- Operation ID: {op_id}")

            if isinstance(params, list) and params:
                ep_lines.append("- Parameters:")
                for p in params:
                    if isinstance(p, dict):
                        p_name = p.get("name", "param")
                        p_in = p.get("in", "query")
                        p_req = "required" if p.get("required") else "optional"
                        p_desc = p.get("description", "")
                        desc_part = f" — {p_desc}" if p_desc else ""
                        ep_lines.append(f"  - {p_name} ({p_in}, {p_req}){desc_part}")

            if isinstance(responses, dict) and responses:
                ep_lines.append("- Responses:")
                for status_code, r_obj in responses.items():
                    r_desc = (
                        r_obj.get("description", "") if isinstance(r_obj, dict) else str(r_obj)
                    )
                    ep_lines.append(f"  - HTTP {status_code}: {r_desc}")

            ep_text = "\n".join(ep_lines)
            lines.append(ep_text)
            lines.append("")
            sections.append(
                {
                    "title": f"{method_upper} {path}",
                    "content": ep_text,
                }
            )

        full_content = "\n".join(lines)
        return full_content, title, sections, total_endpoints

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Decode and parse JSON content into structured document representation."""
        text_content: str
        encoding = "utf-8"
        try:
            text_content = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                encoding = "latin-1"
                text_content = file_bytes.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise ParsingError(
                    filename=filename, reason=f"JSON decoding failed: {exc}"
                ) from exc

        try:
            data = json.loads(text_content)
        except json.JSONDecodeError as exc:
            raise ParsingError(filename=filename, reason=f"Invalid JSON syntax: {exc}") from exc

        derived_title = Path(filename).stem.replace("_", " ").title()

        # Semantic OpenAPI / Swagger parsing
        if self._is_openapi(data):
            openapi_content, openapi_title, openapi_sections, openapi_items = (
                self._parse_openapi(data, derived_title)
            )
            return ParsedDocument(
                content=openapi_content,
                title=openapi_title,
                author=None,
                sections=openapi_sections,
                metadata={
                    "format": "application/json",
                    "document_subtype": "openapi",
                    "encoding": encoding,
                    "root_type": "openapi_spec",
                    "item_count": openapi_items,
                    "byte_size": len(file_bytes),
                },
            )

        sections: list[dict[str, str]] = []
        lines: list[str] = [f"# {derived_title} (JSON Document)"]

        if isinstance(data, list):
            lines.append(f"Total Items: {len(data)}")
            lines.append("")
            batch_lines: list[str] = []
            batch_size = 25

            for idx, item in enumerate(data, start=1):
                if isinstance(item, dict):
                    flat = self._flatten_dict(item)
                    item_str = f"Item {idx}: " + " | ".join(f"{k}: {v}" for k, v in flat.items())
                else:
                    item_str = f"Item {idx}: {item}"
                lines.append(item_str)
                batch_lines.append(item_str)

                if idx % batch_size == 0 or idx == len(data):
                    section_title = (
                        f"{derived_title} (Items {max(1, idx - len(batch_lines) + 1)}-{idx})"
                    )
                    sections.append(
                        {
                            "title": section_title,
                            "content": "\n".join(batch_lines),
                        }
                    )
                    batch_lines = []

            item_count = len(data)
            root_type = "array"

        elif isinstance(data, dict):
            flat = self._flatten_dict(data)
            item_count = len(flat)
            root_type = "object"
            lines.append("## Configuration / Object Fields")
            field_lines = [f"{k}: {v}" for k, v in flat.items()]
            lines.extend(field_lines)
            sections.append(
                {
                    "title": f"{derived_title} Fields",
                    "content": "\n".join(field_lines),
                }
            )
        else:
            item_count = 1
            root_type = type(data).__name__
            val_str = str(data)
            lines.append(val_str)
            sections.append(
                {
                    "title": derived_title,
                    "content": val_str,
                }
            )

        full_content = "\n".join(lines)

        return ParsedDocument(
            content=full_content,
            title=derived_title,
            author=None,
            sections=sections,
            metadata={
                "format": "application/json",
                "encoding": encoding,
                "root_type": root_type,
                "item_count": item_count,
                "byte_size": len(file_bytes),
            },
        )
