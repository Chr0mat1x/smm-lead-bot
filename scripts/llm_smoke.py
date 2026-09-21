"""Проверка LLM-клиентов против настоящего HTTP-сервера.

Поднимаем локальный сервер, который отвечает в формате OpenAI и Anthropic,
и смотрим, что клиенты правильно разбирают ответ и вызовы инструментов.
Моков нет — сетевой слой работает по-настоящему.

    python -m scripts.llm_smoke
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.llm import AnthropicLLM, OpenAICompatibleLLM  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # тишина в консоли
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/chat/completions"):
            has_tools = bool(body.get("tools"))
            if has_tools:
                payload = {
                    "choices": [{"message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "reject_lead",
                                         "arguments": '{"lead_key": "osm:node/1", "reason": "тест"}'},
                        }],
                    }}],
                }
            else:
                payload = {"choices": [{"message": {"content": "Готово."}}]}
        else:  # /v1/messages (Anthropic)
            payload = {"content": [
                {"type": "text", "text": "Смотрю базу."},
                {"type": "tool_use", "id": "tu_1", "name": "show_leads", "input": {"limit": 3}},
            ]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    tools = [{"type": "function", "function": {"name": "reject_lead",
                                              "description": "x", "parameters": {"type": "object"}}}]

    openai = OpenAICompatibleLLM("test-key", base, "test-model")
    reply = openai.chat([{"role": "user", "content": "не тот"}], tools=tools)
    assert reply.wants_tools, "ожидался вызов инструмента"
    assert reply.tool_calls[0].name == "reject_lead"
    assert reply.tool_calls[0].arguments["lead_key"] == "osm:node/1"
    print("OpenAI-совместимый: вызов инструмента разобран верно")

    plain = openai.chat([{"role": "user", "content": "привет"}])
    assert plain.text == "Готово." and not plain.wants_tools
    print("OpenAI-совместимый: текстовый ответ разобран верно")

    claude = AnthropicLLM("test-key", "test-model", base_url=base)
    reply = claude.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "покажи"}], tools=tools)
    assert reply.text == "Смотрю базу."
    assert reply.tool_calls[0].name == "show_leads"
    assert reply.tool_calls[0].arguments == {"limit": 3}
    print("Anthropic: текст и вызов инструмента разобраны верно")

    server.shutdown()
    print("Все проверки LLM прошли.")


if __name__ == "__main__":
    main()
