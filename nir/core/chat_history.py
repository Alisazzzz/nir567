#Chat History class and all stuff about chat history in context



#--------------------------
#---------imports----------
#--------------------------

import json
import re
import os
from typing import Dict, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from nir.prompts import retrieval_prompts



#--------------------------
#-----additional stuff-----
#--------------------------

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chunks = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    token_count = 0
    for chunk in chunks:
        length = len(chunk)
        if re.match(r"[A-Za-z0-9]+$", chunk):
            token_count += max(1, round(length / 3.5))
        elif re.match(r"[А-Яа-яЁё]+$", chunk):
            token_count += max(1, round(length / 2.4))
        else:
            token_count += 1
    return token_count

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



#---------------------------------------
#-----checking if the topic changed-----
#---------------------------------------

class TopicCheckResult(BaseModel):
    is_new_topic: bool
    summary: Optional[str]

topic_parser = PydanticOutputParser(pydantic_object=TopicCheckResult)

prompt_topic_check = ChatPromptTemplate.from_messages([
    ("system", retrieval_prompts.SYSTEM_PROMPT_TOPIC_CHECK),
    ("human", "Current Context Summary: {current_summary}\n\nUser's new message:\n{new_message}")
]).partial(format_instructions=topic_parser.get_format_instructions())



#----------------------------
#-----Chat History Class-----
#----------------------------

class ChatHistory:

    def __init__(self, graph_path: str, file_path: Optional[str] = None):
        self.graph_path = graph_path
        self.file_path = file_path
        self.messages: List[Dict[str, str]] = []  # [{"role": "user/assistant", "content": "..."}]
        self.current_topic_summary: Optional[str] = None 


    def add_message_to_history(self, role: str, content: str):
        if role not in ["user", "assistant"]:
            return
        self.messages.append({"role": role, "content": content})


    def save(self, path: Optional[str] = None):
        save_path = path if path else self.file_path
        if not save_path:
            return
        data = {
            "graph_path": self.graph_path,
            "current_topic_summary": self.current_topic_summary,
            "messages": self.messages
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.file_path = save_path


    @classmethod
    def load(cls, path: str) -> "ChatHistory":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(graph_path=data["graph_path"], file_path=path)
        instance.messages = data.get("messages", [])
        instance.current_topic_summary = data.get("current_topic_summary")
        return instance


    def _calculate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += estimate_tokens(msg.get("content", ""))
        return total


    def get_context_window(self, max_tokens: int = 1024) -> str:
        if not self.messages:
            return ""
        selected_messages = []
        current_tokens = 0
        for msg in reversed(self.messages):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens <= max_tokens:
                selected_messages.append(msg)
                current_tokens += msg_tokens
            else:
                break
        selected_messages.reverse()
        result = "CHAT HISTORY:\n\n"
        for msg in selected_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            result += f"{role}: {msg['content']}\n"
        return result


    def get_context_with_topic_detection(
            self, 
            new_user_message: str, 
            llm: BaseLanguageModel, 
            max_tokens: int = 1024
        ) -> str:

        chain_topic = prompt_topic_check | llm | clean_json | topic_parser
        try:
            result_topic = chain_topic.invoke({
                "current_summary": self.current_topic_summary if self.current_topic_summary else "No previous topic.",
                "new_message": new_user_message
            })
            if result_topic.is_new_topic and result_topic.summary:
                self.current_topic_summary = result_topic.summary
                print(f"Topic updated: {self.current_topic_summary}")
            else:
                print("Topic continuation detected.")         
        except Exception as e:
            print(f"Error during topic detection: {e}")

        context_parts = []
        if self.current_topic_summary:
            topic_block = f"CURRENT CONVERSATION TOPIC:\n{self.current_topic_summary}\n\n"
            context_parts.append(topic_block)
            max_tokens -= estimate_tokens(topic_block)

        temp_messages = self.messages + [{"role": "user", "content": new_user_message}]
        selected_messages = []
        current_tokens = 0
        for msg in reversed(temp_messages):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens <= max_tokens:
                selected_messages.append(msg)
                current_tokens += msg_tokens
            else:
                break
        selected_messages.reverse()
        history_block = "CHAT HISTORY:\n\n"
        for msg in selected_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_block += f"{role}: {msg['content']}\n"
        context_parts.append(history_block)
        return "".join(context_parts)