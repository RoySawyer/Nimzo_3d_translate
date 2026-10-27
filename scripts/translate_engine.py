import os
import re
import sys
import argparse
from typing import List

import polib
from openai import OpenAI

TOKEN_REGEX = re.compile(r"(%[sfd]|\{[a-zA-Z0-9_]+\}|<[^>]+>|\[/?[a-zA-Z0-9]+\])")
LANG_CODE_REGEX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
DIVIDER = "---DIVIDER---"


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


def merge_existing_translations(po, output_path: str):
    """Keep translations from an earlier catalog and add entries from the current POT."""
    if not os.path.isfile(output_path):
        return po

    try:
        previous = polib.pofile(output_path)
    except Exception as err:
        raise SystemExit(f"ERROR: cannot read existing output catalog '{output_path}': {err}")

    previous_by_key = {
        (entry.msgctxt, entry.msgid): entry
        for entry in previous
        if entry.msgid
    }
    for entry in po:
        if not entry.msgid:
            continue
        old_entry = previous_by_key.get((entry.msgctxt, entry.msgid))
        if old_entry is None:
            continue
        entry.msgstr = old_entry.msgstr
        if "fuzzy" in old_entry.flags and "fuzzy" not in entry.flags:
            entry.flags.append("fuzzy")
        elif "fuzzy" not in old_entry.flags and "fuzzy" in entry.flags:
            entry.flags.remove("fuzzy")
    return po


def load_catalog(pot_path: str, output_path: str):
    if not os.path.isfile(pot_path):
        raise SystemExit(f"ERROR: POT template not found: {pot_path}")
    try:
        po = polib.pofile(pot_path)
    except Exception as err:
        raise SystemExit(f"ERROR: cannot parse POT template '{pot_path}': {err}")

    if any(entry.msgid_plural for entry in po):
        raise SystemExit("ERROR: plural entries are not supported by this translation engine yet.")
    return merge_existing_translations(po, output_path)


def translate_batch(client: OpenAI, model: str, texts: List[str], target_lang: str) -> List[str]:
    prompt = (
        f"You are a professional software localizer for a 3D Chess Application.\n"
        f"Translate the following list of UI strings to {target_lang}.\n"
        f"Rules:\n"
        f"1. Preserve all placeholders exactly: %s, %d, {{var}}, <tags>, [tags].\n"
        f"2. Retain exact newlines and whitespace formatting.\n"
        f"3. Return ONLY the translated strings, separated by '{DIVIDER}'.\n"
        f"4. Do not output numbering, commentary, quotes, or markdown wrappers."
    )
    payload = f"\n{DIVIDER}\n".join(texts)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    if not isinstance(raw, str):
        raise RuntimeError("translation API returned an empty or invalid response")
    return [text.strip() for text in raw.strip().split(DIVIDER)]


def validate_complete_catalog(po):
    unresolved = [
        entry for entry in po
        if entry.msgid and (not entry.msgstr.strip() or "fuzzy" in entry.flags)
    ]
    if unresolved:
        samples = ", ".join(repr(entry.msgid[:60]) for entry in unresolved[:3])
        raise RuntimeError(
            f"Catalog is incomplete: {len(unresolved)} untranslated or fuzzy entries remain. "
            f"Examples: {samples}"
        )


def main():
    args = parse_args()
    if not args.lang.strip():
        raise SystemExit("ERROR: target language name cannot be empty")
    if not LANG_CODE_REGEX.fullmatch(args.code):
        raise SystemExit("ERROR: language code may contain only letters, numbers, '_' and '-'")

    po = load_catalog(args.pot, args.output)
    pending = [
        entry for entry in po
        if entry.msgid and (not entry.msgstr.strip() or "fuzzy" in entry.flags)
    ]
    po.metadata["Language"] = args.code

    if pending:
        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise SystemExit("ERROR: MODEL_API_KEY environment variable not set.")
        api_base = os.getenv("MODEL_API_BASE", "https://api.cerebras.ai/v1")
        model_name = os.getenv("MODEL_NAME", "llama3.1-8b")
        client = OpenAI(api_key=api_key, base_url=api_base)

        output_dir = os.path.dirname(args.output) or "."
        os.makedirs(output_dir, exist_ok=True)
        po.save(args.output)
        batch_size = 15
        print(f"Beginning translation for [{args.lang}] ({len(pending)} strings pending)...")

        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            batch_number = i // batch_size + 1
            try:
                translations = translate_batch(client, model_name, [e.msgid for e in batch], args.lang)
                if len(translations) != len(batch):
                    print(f"Count mismatch in batch {batch_number}; retrying its entries individually.")
                    translations = []
                    for entry in batch:
                        single = translate_batch(client, model_name, [entry.msgid], args.lang)
                        if len(single) != 1:
                            raise RuntimeError("individual retry did not return exactly one translation")
                        translations.append(single[0])

                for position, (entry, translation) in enumerate(zip(batch, translations), start=1):
                    if not translation.strip():
                        raise RuntimeError(f"empty translation at position {position} in batch {batch_number}")
                    if not validate_format_parity(entry.msgid, translation):
                        raise RuntimeError(f"placeholder mismatch at position {position} in batch {batch_number}")
                    entry.msgstr = translation
                    if "fuzzy" in entry.flags:
                        entry.flags.remove("fuzzy")
            except Exception as err:
                po.save(args.output)
                raise RuntimeError(
                    f"Batch {batch_number} failed; completed work was checkpointed to '{args.output}': {err}"
                ) from err

            po.save(args.output)
            print(f"Progress: {min(i + batch_size, len(pending))}/{len(pending)} pending strings processed.")
    else:
        print("No untranslated strings found; validating the existing catalog.")

    validate_complete_catalog(po)
    po.metadata["Language"] = args.code
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    po.save(args.output)
    print(f"Completed! Output catalog saved to: {args.output}")


if __name__ == "__main__":
    main()
