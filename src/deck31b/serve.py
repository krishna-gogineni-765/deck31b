"""Serve deck31b as a TypeSafe-compatible FP8 endpoint.

The weights are the frozen public checkpoint google/gemma-4-31B-it. This
process does not merge an adapter. At startup it quantizes linear weights
and activations to FP8 e4m3, then scores each question from the next-token
distribution. No tokens are generated.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch

BASE_MODEL = "google/gemma-4-31B-it"
BASE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
TEMPERATURE = 3.4
MAX_CONTEXT = 16_384
NAME = "deck31b"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)


def options_from(criteria: Any, qtype: str) -> list[tuple[str, str, Any]]:
    if qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must list at least two levels")
        items = [(str(i), value) for i, value in enumerate(criteria)]
    else:
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if qtype == "noul" and not isinstance(criteria, dict):
            criteria = {
                "true": "The proposition is true.",
                "false": "The proposition is false.",
            }
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice criteria must name at least two options")
        items = list(criteria.items())
    if len(items) > len(LETTERS):
        raise ValueError(f"{len(items)} options exceeds the 26-letter alphabet")
    return [(LETTERS[i], key, value) for i, (key, value) in enumerate(items)]


def build_prompt(state: Any, instructions: Any, opts: list[tuple[str, str, Any]]) -> str:
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    lines = [state_text.rstrip(), "", instructions.rstrip(), "", "Options:"]
    for letter, _label, description in opts:
        lines.append(f"{letter}. {description}")
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    return "\n".join(lines)


def letter_token_ids(tokenizer) -> dict[str, list[int]]:
    ids: dict[str, list[int]] = defaultdict(list)
    for index in range(len(tokenizer)):
        text = tokenizer.decode([index])
        if len(text) == 1 and text in LETTERS:
            ids[text].append(index)
    missing = [letter for letter in LETTERS if not ids[letter]]
    if missing:
        raise RuntimeError(f"tokenizer has no single-character token for {missing[:4]}")
    return dict(ids)


def _temper(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return probs
    scaled = np.maximum(probs, 1e-12) ** (1.0 / temperature)
    return scaled / scaled.sum()


class Deck31B:
    def __init__(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        quantize_(self.model, Float8DynamicActivationFloat8WeightConfig(set_inductor_config=False))
        torch.cuda.empty_cache()
        self.letter_ids = letter_token_ids(self.tokenizer)
        self._warmup()

    def _warmup(self) -> None:
        question = {
            "type": "noul",
            "instructions": "Is the statement true?",
            "criteria": {"true": "Yes.", "false": "No."},
        }
        self.probabilities("ok", question)

    @torch.inference_mode()
    def probabilities(self, state: Any, question: dict) -> tuple[dict[str, float], int]:
        kind = question.get("type")
        if kind not in ("noul", "choice", "score"):
            raise ValueError(f"unknown question type {kind!r}")
        if "instructions" not in question:
            raise ValueError("question is missing instructions")
        opts = options_from(question.get("criteria"), kind)
        prompt = build_prompt(state, question.get("instructions") or "", opts)
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        count = int(ids.shape[1])
        if count > MAX_CONTEXT:
            raise ValueError(f"prompt has {count} tokens; maximum is {MAX_CONTEXT}")
        logits = self.model(input_ids=ids.to(self.model.device)).logits[0, -1].float()
        full = torch.softmax(logits, dim=-1)
        raw = np.array(
            [sum(float(full[token]) for token in self.letter_ids[letter]) for letter, _label, _desc in opts],
            dtype=np.float64,
        )
        total = float(raw.sum())
        if total <= 0.0:
            raise ValueError("letter probability mass is zero")
        raw /= total
        tempered = _temper(raw, TEMPERATURE)
        labels = [str(label) for _letter, label, _desc in opts]
        return {label: float(value) for label, value in zip(labels, tempered, strict=True)}, count


def answer(question: dict, probs: dict[str, float], tokens: int) -> dict:
    kind = question["type"]
    out = {"type": kind, "confidence": max(probs.values()), "input_tokens": tokens}
    if kind == "noul":
        out["noul"] = probs["true"]
    elif kind == "choice":
        out.update(choice=max(probs, key=probs.get), probabilities=probs)
    else:
        out.update(score=sum(int(key) * value for key, value in probs.items()), probabilities=probs)
    return out


def make_handler(model: Deck31B):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": NAME})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            started = time.perf_counter()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)))
                questions = body["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
                answers = {}
                with lock:
                    for qid, question in questions.items():
                        probs, tokens = model.probabilities(body["state"], question)
                        answers[qid] = answer(question, probs, tokens)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
                return
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._send(500, {"error": "CUDA out of memory"})
                return
            tokens = sum(item.pop("input_tokens") for item in answers.values())
            self._send(
                200,
                {
                    "model": body.get("model") or NAME,
                    "answers": answers,
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "latency_ms": round((time.perf_counter() - started) * 1e3, 2),
                },
            )

        def log_message(self, *_args) -> None:
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("deck31b requires an NVIDIA GPU with FP8 support")
    model = Deck31B()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    print(f"serving {NAME} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
