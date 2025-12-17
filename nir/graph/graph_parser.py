#All stuff for safe graph structures parsing if need to preserve strange data is here



#--------------------------
#---------imports----------
#--------------------------

import json
import re



#--------------------------
#-----additional stuff-----
#--------------------------

def remove_comments(s: str) -> str:
    out_chars = []
    i = 0
    n = len(s)
    in_string = False
    string_quote = ""
    in_single_line_comment = False
    in_multi_line_comment = False
    while i < n:
        c = s[i]

        if in_single_line_comment:
            if c == "\n":
                in_single_line_comment = False
                out_chars.append(c)
            i += 1
            continue
        
        if in_multi_line_comment:
            if c == "*" and i + 1 < n and s[i + 1] == "/":
                in_multi_line_comment = False
                i += 2
            else:
                i += 1
            continue
        
        if in_string:
            if c == "\\":
                if i + 1 < n:
                    out_chars.append(c)
                    out_chars.append(s[i + 1])
                    i += 2
                else:
                    out_chars.append(c)
                    i += 1
                continue
            elif c == string_quote:
                out_chars.append(c)
                in_string = False
                string_quote = ""
                i += 1
                continue
            else:
                out_chars.append(c)
                i += 1
                continue

        if c == '"' or c == "'":
            in_string = True
            string_quote = c
            out_chars.append(c)
            i += 1
            continue

        if c == "/" and i + 1 < n and s[i + 1] == "/":
            in_single_line_comment = True
            i += 2
            continue

        if c == "/" and i + 1 < n and s[i + 1] == "*":
            in_multi_line_comment = True
            i += 2
            continue

        if c == "#":
            prev = s[i - 1] if i - 1 >= 0 else "\n"
            if prev in {"\n", "\r", "\t", " ", ""}:
                in_single_line_comment = True
                i += 1
                continue
            else:
                out_chars.append(c)
                i += 1
                continue

        out_chars.append(c)
        i += 1
    return "".join(out_chars)

def extract_last_json(text: str) -> str:
    stack = 0
    start = None
    last = None
    for i, ch in enumerate(text):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    last = text[start:i+1]
    return last

def clean_json(text: str) -> str:
    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if codeblock_match:
        possible_json = codeblock_match.group(1).strip()
        cleaned = remove_comments(possible_json)
        return cleaned

    balanced = extract_last_json(text)
    if balanced:
        try:
            json.loads(balanced)
            cleaned = remove_comments(balanced)
            return cleaned
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r"^[^{]+", "", text)
    cleaned = re.sub(r"[^}]+$", "", cleaned)
    cleaned = remove_comments(cleaned)
    cleaned = re.sub(r'(":\s*"[^"]*")\s*\([^)]*\)', r'\1', cleaned)
    return cleaned

def safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {}



#------------------------------------
#-----functions to ensure values-----
#------------------------------------

def get_value(data: dict, key: str, default):
    if isinstance(data, dict) and key in data:
        return data[key]
    return default

def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def ensure_dict(value):
    if isinstance(value, dict):
        return value
    return {}

def ensure_str(value):
    if value is None:
        return ""
    return str(value)

def ensure_float(value, default=1.0):
    try:
        return float(value)
    except Exception:
        return default



#---------------------------------------------------------------------
#-----normalize ouput structures: first stage of graph extraction-----
#---------------------------------------------------------------------

def normalize_extracted_node(raw: dict) -> dict:
    return {
        "name": ensure_str(get_value(raw, "name", "")),
        "type": ensure_str(get_value(raw, "type", "item")),
        "base_description": ensure_str(get_value(raw, "base_description", "")),
        "base_attributes": ensure_dict(get_value(raw, "base_attributes", {})),
    }

def normalize_extracted_edge(raw: dict) -> dict:
    return {
        "node1": get_value(raw, "node1", None),
        "node2": get_value(raw, "node2", None),
        "relation_from1to2": ensure_str(
            get_value(raw, "relation_from1to2", get_value(raw, "relation", "related_to"))
        ),
        "relation_from2to1": ensure_str(
            get_value(raw, "relation_from2to1", "related_to")
        ),
        "description": ensure_str(get_value(raw, "description", "")),
        "weight": ensure_float(get_value(raw, "weight", 1.0)),
    }

def normalize_graph_extraction_result(raw: dict) -> dict:
    raw_nodes = ensure_list(get_value(raw, "nodes", []))
    raw_edges = ensure_list(get_value(raw, "edges", []))
    return {
        "nodes": [normalize_extracted_node(n) for n in raw_nodes if isinstance(n, dict)],
        "edges": [normalize_extracted_edge(e) for e in raw_edges if isinstance(e, dict)],
    }



#---------------------------------------------------------------------
#-----normalize ouput structures: semi stage of graph extraction------
#---------------------------------------------------------------------

def normalize_merged_node(raw: dict) -> dict:
    return {
        "name": get_value(raw, "name", None),
        "base_description": ensure_str(get_value(raw, "base_description", "")),
        "base_attributes": ensure_dict(get_value(raw, "base_attributes", {})),
    }



#-----------------------------------------------------------------------
#-----normalize ouput structures: second stage of graph extraction------
#-----------------------------------------------------------------------

def normalize_affected_node(raw: dict) -> dict:
    return {
        "id": ensure_str(get_value(raw, "id", "")),
        "name": ensure_str(get_value(raw, "name", "")),
        "new_current_description": ensure_str(
            get_value(raw, "new_current_description",
                      get_value(raw, "description", ""))
        ),
        "new_current_attributes": ensure_dict(
            get_value(raw, "new_current_attributes", {})
        ),
        "time_start_event": get_value(raw, "time_start_event", None),
        "time_end_event": get_value(raw, "time_end_event", None),
    }

def normalize_affected_edge(raw: dict) -> dict:
    return {
        "id": ensure_str(get_value(raw, "id", "")),
        "new_description": ensure_str(
            get_value(raw, "new_description",
                      get_value(raw, "description", ""))
        ),
        "time_start_event": get_value(raw, "time_start_event", None),
        "time_end_event": get_value(raw, "time_end_event", None),
    }

def normalize_event_impact(raw: dict) -> dict:
    return {
        "event_name": ensure_str(get_value(raw, "event_name", "")),
        "affected_nodes": [
            normalize_affected_node(n)
            for n in ensure_list(get_value(raw, "affected_nodes", []))
            if isinstance(n, dict)
        ],
        "affected_edges": [
            normalize_affected_edge(e)
            for e in ensure_list(get_value(raw, "affected_edges", []))
            if isinstance(e, dict)
        ],
    }

def normalize_events_subgraph(raw: dict) -> dict:
    events = ensure_list(get_value(raw, "events_with_impact", get_value(raw, "events", [])))
    return {
        "events_with_impact": [
            normalize_event_impact(e)
            for e in events
            if isinstance(e, dict)
        ]
    }



#--------------------------------
#-----class for safe parser------
#--------------------------------

class SafePydanticParser:
    def __init__(self, expected_structure, normalizer):
        self.expected_structure = expected_structure
        self.normalizer = normalizer

    def parse(self, text: str):
        raw_json = clean_json(text)
        data = safe_json_loads(raw_json)
        normalized = self.normalizer(data)
        return self.expected_structure.parse_obj(normalized)
