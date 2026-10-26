import os
import re
import sys
import argparse
from typing import List
import polib
from openai import OpenAI

TOKEN_REGEX = re.compile(r"(%[sfd]|\{[a-zA-Z0-9_]+\}|<[^>]+>|\[/?[a-zA-Z0-9]+\])")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pot", required=True, help="Path to input .pot/.po file")
    parser.add_argument("--lang", required=True, help="Target language name")
    parser.add_argument("--code", required=True, help="Target language code")
    parser.add_argument("--output", required=True, help="Path to write output .po")
    return parser.parse_args()

def validate_format_parity(src: str, tgt: str) -> bool:
    src_tokens = sorted(TOKEN_REGEX.findall(src))
    tgt_tokens = sorted(TOKEN_REGEX.findall(tgt))
    return src_tokens == tgt_tokens

def translate_batch(client: OpenAI, model: str, texts: List[str], target_lang: str) -> List[str]:
    prompt = (
        f"You are a professional software localizer for a 3D Chess Application.\n"
        f"Translate the following list of UI strings to {target_lang}.\n"
        f"Rules:\n"
        f"1. Preserve all placeholders exactly: %s, %d, {{var}}, <tags>, [tags].\n"
        f"2. Retain exact newlines and whitespace formatting.\n"
        f"3. Return ONLY the translated strings, separated by '---DIVIDER---'.\n"
        f"4. Do not output numbering, commentary, quotes, or markdown wrappers."
    )
    payload = "\n---DIVIDER---\n".join(texts)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload}
        ],
        temperature=0.1
    )
    
    raw = response.choices[0].message.content.strip()
    return [s.strip() for s in raw.split("---DIVIDER---")]

def main():
    args = parse_args()
    api_key = os.getenv("MODEL_API_KEY")
    api_base = os.getenv("MODEL_API_BASE", "https://api.cerebras.ai/v1")
    model_name = os.getenv("MODEL_NAME", "llama3.1-8b")

    if not api_key:
        print("ERROR: MODEL_API_KEY environment variable not set.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    po = polib.pofile(args.pot)
    client = OpenAI(api_key=api_key, base_url=api_base)

    entries = [e for e in po if not e.msgstr.strip() or "fuzzy" in e.flags]
    batch_size = 15

    print(f"Beginning translation for [{args.lang}] ({len(entries)} strings pending)...")

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        src_texts = [e.msgid for e in batch]

        try:
            translations = translate_batch(client, model_name, src_texts, args.lang)
            if len(translations) != len(batch):
                print(f"Count mismatch in batch {i // batch_size + 1}. Retrying individual entries...")
                for entry in batch:
                    single = translate_batch(client, model_name, [entry.msgid], args.lang)
                    entry.msgstr = single[0] if single else entry.msgid
            else:
                for entry, tr in zip(batch, translations):
                    if validate_format_parity(entry.msgid, tr):
                        entry.msgstr = tr
                        if "fuzzy" in entry.flags:
                            entry.flags.remove("fuzzy")
                    else:
                        print(f"Format parity mismatch rejected for: '{entry.msgid}' -> '{tr}'")
                        entry.msgstr = entry.msgid
                        entry.flags.append("fuzzy")

        except Exception as err:
            print(f"Batch failed at index {i}: {err}")

        po.save(args.output)
        print(f"Progress: {min(i + batch_size, len(entries))}/{len(entries)} strings processed.")

    po.metadata["Language"] = args.code
    po.save(args.output)
    print(f"Completed! Output catalog saved to: {args.output}")

if __name__ == "__main__":
    main()
