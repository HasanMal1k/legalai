"""
eval/translate.py

Translates the Urdu and Roman Urdu evaluation questions into English with the local model, as
the application would do before searching. The model is told to translate only: no law names
or section numbers that the question does not contain. Writes eval/translations.jsonl.

    python eval/translate.py
"""

import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:12b-it-qat"
PROMPT = """Translate this question from {language} into natural English, the way a Pakistani lawyer would put the same question.
Translate only. Do not answer it, and do not add the name of any law, section or court that the question itself does not mention.
Keep Pakistani legal terms that have no exact English word (such as khula, haq mehr, qisas, hiba) and put their English meaning in brackets after them.

Question: {question}"""
SCHEMA = {"type": "object", "properties": {"en": {"type": "string"}}, "required": ["en"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    questions = [json.loads(line) for line in (HERE / "eval_set.jsonl").open(encoding="utf-8")]
    out_path = HERE / "translations.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for question in questions:
            if question["lang"] not in ("ru", "ur"):
                continue
            language = "Roman Urdu (Urdu written in Latin letters)" if question["lang"] == "ru" else "Urdu"
            resp = requests.post(OLLAMA, json={
                "model": MODEL, "stream": False, "think": False, "format": SCHEMA,
                "messages": [{"role": "user", "content": PROMPT.format(language=language, question=question["q"])}],
                "options": {"temperature": 0, "num_ctx": 2048},
            }, timeout=600)
            resp.raise_for_status()
            english = json.loads(resp.json()["message"]["content"])["en"].strip()
            out.write(json.dumps({"id": question["id"], "lang": question["lang"], "q": question["q"], "en": english}, ensure_ascii=False) + "\n")
            print(f"{question['id']}: {english}")


if __name__ == "__main__":
    main()
