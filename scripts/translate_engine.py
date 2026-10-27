import os
import re
import sys
import time
import argparse
from typing import List

import polib
from openai import OpenAI, RateLimitError, APIError

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
        print(f"Warning: could not read existing output catalog '{output_path}': {err}")
        return po

    previous_by_key = {
        (entry.msgctxt, entry.msgid): entry
        for entry in previous
        if entry.msgid
    }
    for entry in po:
        if not entry.msgid:
            continue
        old_entry = previous_by_key.get((entry.msgctxt, entry.msgid))
        if old_entry is None or not old_entry.msgstr.strip():
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

    return merge_existing_translations(po, output_path)


def translate_with_backoff(client: OpenAI, model: str, prompt: str, user_content: str, max_retries: int = 5):
    """Calls Cerebras API with exponential backoff on 429/RateLimit."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIError) as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Rate limited or API glitch ({e}). Backing off {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2.0
        except Exception as e:
            raise e


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
    raw = translate_with_backoff(client, model, prompt, payload)
    return [text.strip() for text in raw.strip().split(DIVIDER)]


def translate_single_fallback(client: OpenAI, model: str, text: str, target_lang: str) -> str:
    """Fallback when batch delimiter or tokens fail."""
    prompt = (
        f"You are a professional software localizer for a 3D Chess Application.\n"
        f"Translate this exact string into {target_lang}.\n"
        f"Preserve all formatting and placeholders (%s, %d, {{var}}, tags) unchanged.\n"
        f"Return ONLY the direct translation."
    )
    raw = translate_with_backoff(client, model, prompt, text)
    return raw.strip().strip('"').strip("'")


def set_language_metadata(po, language_code: str):
    po.metadata["Language"] = language_code
    for entry in po:
        if not entry.msgid and "fuzzy" in entry.flags:
            entry.flags.remove("fuzzy")


def main():
    args = parse_args()
    if not args.lang.strip():
        raise SystemExit("ERROR: target language name cannot be empty")
    if not LANG_CODE_REGEX.fullmatch(args.code):
        raise SystemExit("ERROR: language code may contain only letters, numbers, '_' and '-'")

    po = load_catalog(args.pot, args.output)
    set_language_metadata(po, args.code)

    pending = [
        entry for entry in po
        if entry.msgid and (not entry.msgstr.strip() or "fuzzy" in entry.flags)
    ]

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    if pending:
        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise SystemExit("ERROR: MODEL_API_KEY environment variable not set.")
        api_base = os.getenv("MODEL_API_BASE", "https://api.cerebras.ai/v1")
        model_name = os.getenv("MODEL_NAME", "llama3.1-8b")
        client = OpenAI(api_key=api_key, base_url=api_base)

        batch_size = 15
        print(f"Beginning translation for [{args.lang}] ({len(pending)} strings pending)...")

        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            batch_number = i // batch_size + 1
            
            try:
                translations = translate_batch(client, model_name, [e.msgid for e in batch], args.lang)
                if len(translations) != len(batch):
                    translations = [translate_single_fallback(client, model_name, e.msgid, args.lang) for e in batch]

                for entry, translation in zip(batch, translations):
                    if not translation.strip():
                        translation = translate_single_fallback(client, model_name, entry.msgid, args.lang)

                    # Validate placeholders
                    if validate_format_parity(entry.msgid, translation):
                        entry.msgstr = translation
                        if "fuzzy" in entry.flags:
                            entry.flags.remove("fuzzy")
                    else:
                        # Fallback attempt if placeholders were mangled
                        single = translate_single_fallback(client, model_name, entry.msgid, args.lang)
                        if validate_format_parity(entry.msgid, single):
                            entry.msgstr = single
                            if "fuzzy" in entry.flags:
                                entry.flags.remove("fuzzy")
                        else:
                            # Keep untranslated or marked fuzzy so the run doesn't fail
                            print(f"Warning: Placeholder parity could not be satisfied for '{entry.msgid[:30]}...'")
                            entry.msgstr = entry.msgid
                            if "fuzzy" not in entry.flags:
                                entry.flags.append("fuzzy")

            except Exception as err:
                po.save(args.output)
                raise RuntimeError(
                    f"Batch {batch_number} failed; progress saved to '{args.output}': {err}"
                ) from err

            # Save checkpoint every batch
            po.save(args.output)
            print(f"Progress: {min(i + batch_size, len(pending))}/{len(pending)} strings processed.")
            # Be polite to Cerebras rate limiter
            time.sleep(1.0)
    else:
        print("All strings are already translated.")

    po.metadata["Language"] = args.code
    po.save(args.output)
    print(f"Completed! Output catalog saved to: {args.output}")


if __name__ == "__main__":
    main()
