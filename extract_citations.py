#!/usr/bin/env python3
"""
Extract citations from all Likkutei Sichos footnotes using Claude Haiku.
Produces citations.json — the master citation database.

Each citation:
  {
    "sicha_id": "1_1",          # page_volume
    "volume": 1,
    "page": 1,
    "title": "בראשית",
    "fn": 9,                    # footnote number
    "raw": "קידושין מ, ב",      # exact text span from footnote
    "work": "Kiddushin",        # normalized Sefaria book key
    "ref": "Kiddushin.40b",     # Sefaria ref
    "category": "bavli"         # tanach|mishnah|bavli|yerushalmi|midrash|zohar|rambam|tanya|chassidus|other
  }
"""

import json
import os
import re
import time
from pathlib import Path

BASE_URL = "http://host.docker.internal:3001"
PROGRESS_PATH = Path("citations_progress.json")
OUT_PATH = Path("citations.json")
LS_DIR = Path("ls_data")

HTML_TAG_RE = re.compile(r'<[^>]+>')

def strip_html(text):
    return HTML_TAG_RE.sub('', text).strip()

def get_api_key():
    return (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or
            os.environ.get("ANTHROPIC_API_KEY") or "dummy")

def make_client():
    import anthropic
    return anthropic.Anthropic(api_key=get_api_key(), base_url=BASE_URL)

SYSTEM_PROMPT = """You are an expert in rabbinic literature. Extract all source citations from Hebrew/Yiddish footnotes.

For each footnote paragraph, identify every citable source — texts the author is pointing to.
Return ONLY a JSON array. Each element:
{
  "fn": <footnote number as integer>,
  "raw": "<exact substring from footnote text that is the citation>",
  "work": "<English Sefaria book name, e.g. Kiddushin, Genesis, Psalms, Zohar, Tanya, Torah Ohr, Likkutei Torah>",
  "ref": "<Sefaria-style ref, e.g. Kiddushin.40b, Genesis.1.1, Zohar.2.161a, Tanya.2.2, Torah Ohr, Lech Lecha 11, Likkutei Torah, Vayikra 41>",
  "category": "<one of: tanach|mishnah|bavli|yerushalmi|midrash|zohar|rambam|tanya|chassidus|other>"
}

RULES:
- Include: Tanach, Talmud, Mishnah, Zohar, Rambam, Tanya, Torah Ohr (תורה אור), Likkutei Torah (לקוטי תורה), Siddur
- Exclude: ספר המאמרים, מאמרים, Likkutei Sichos itself, footnote cross-references (like "ראה הערה X"), responsa
- For Talmud Bavli: ref = BookName.Xb (e.g. Kiddushin.40b, Berakhot.2a)
- For Tanach: ref = Book.chapter.verse (e.g. Genesis.1.1, Psalms.119.89)
- For Mishnah: ref = Mishnah_BookName.chapter.mishnah (e.g. Mishnah_Avot.3.1)
- For Zohar: זח"א = Zohar.1, זח"ב = Zohar.2, זח"ג = Zohar.3; ref = Zohar.volume.page (e.g. Zohar.2.161a)
- For Tanya: ח"א/ליקוטי אמרים = Tanya part 1; ref = Tanya.part.chapter (e.g. Tanya.1.3)
- For Torah Ohr (ת"א/תורה אור): category=chassidus; ref = "Torah Ohr, Parsha PageNum" (e.g. "Torah Ohr, Lech Lecha 11")
- For Likkutei Torah (לקו"ת/ל"ת): category=chassidus; ref = "Likkutei Torah, Parsha PageNum" (e.g. "Likkutei Torah, Vayikra 41")
- For Rambam/Mishneh Torah: ref = "Mishneh Torah, Hilchot X.chapter.law"
- "raw" must be the EXACT text span from the input that contains the citation
- Return one element per citation; return [] if none
- Return ONLY the JSON array, no explanation"""

def extract_citations_for_sicha(client, sicha):
    """Call Haiku to extract citations from this sicha's footnotes."""
    footnotes_html = sicha.get('footnotes', '')
    if not footnotes_html or not footnotes_html.strip():
        return []

    # Strip HTML for LLM but keep footnote numbers
    # Convert <p>N) text...</p> to plain text
    text = strip_html(footnotes_html)
    if not text.strip():
        return []

    prompt = f"""Extract all source citations from these footnotes (sicha: {sicha['title']}, vol {sicha['volume']}):

{text[:4000]}"""

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_resp = resp.content[0].text.strip()
            # If response was truncated mid-JSON, try to salvage it
            if not raw_resp.endswith(']') and '[' in raw_resp:
                # Find the last complete object
                try:
                    last_brace = raw_resp.rfind('}')
                    if last_brace > 0:
                        raw_resp = raw_resp[:last_brace+1] + ']'
                except Exception:
                    pass

            # Parse JSON
            # Sometimes the model wraps in ```json ... ```
            if raw_resp.startswith('```'):
                raw_resp = re.sub(r'^```\w*\n?', '', raw_resp)
                raw_resp = re.sub(r'\n?```$', '', raw_resp)

            citations = json.loads(raw_resp)
            if not isinstance(citations, list):
                return []
            return citations

        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
            continue
        except Exception as e:
            err = str(e)
            if "401" in err or "authentication" in err.lower():
                client = make_client()
            if attempt < 2:
                time.sleep(2)
            continue

    return []

def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"done": [], "citations": []}

def save_progress(progress):
    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False))

def main():
    import anthropic

    # Load all sichos
    all_sichos = []
    for vol_file in sorted(LS_DIR.glob("vol_*.json"),
                           key=lambda f: int(f.stem.split('_')[1])):
        vol = int(vol_file.stem.split('_')[1])
        sichos = json.loads(vol_file.read_text())
        for s in sichos:
            s['volume'] = vol
            all_sichos.append(s)

    print(f"Total sichos: {len(all_sichos)}")

    progress = load_progress()
    done_set = set(progress["done"])
    all_citations = progress["citations"]

    pending = [s for s in all_sichos
               if f'{s["page"]}_{s["volume"]}' not in done_set]
    print(f"Done: {len(done_set)}, Pending: {len(pending)}")

    client = make_client()

    for i, sicha in enumerate(pending):
        sicha_id = f'{sicha["page"]}_{sicha["volume"]}'

        citations = extract_citations_for_sicha(client, sicha)

        # Attach sicha context to each citation
        for c in citations:
            if isinstance(c, dict) and 'fn' in c:
                c['sicha_id'] = sicha_id
                c['volume'] = sicha['volume']
                c['page'] = sicha['page']
                c['title'] = sicha.get('title', '')
                all_citations.append(c)

        done_set.add(sicha_id)
        progress["done"].append(sicha_id)
        progress["citations"] = all_citations

        if (i + 1) % 10 == 0:
            save_progress(progress)
            pct = (len(done_set)) / len(all_sichos) * 100
            print(f"[{i+1}/{len(pending)}] {pct:.0f}% — "
                  f"last: vol {sicha['volume']} '{sicha['title']}' "
                  f"→ {len(citations)} citations (total: {len(all_citations)})",
                  flush=True)
        elif (i + 1) % 2 == 0:
            print(f"  [{i+1}/{len(pending)}] vol {sicha['volume']} p{sicha['page']} "
                  f"'{sicha['title']}' → {len(citations)} (total: {len(all_citations)})",
                  flush=True)

        # Refresh client periodically
        if (i + 1) % 50 == 0:
            client = make_client()

        time.sleep(0.3)

    save_progress(progress)

    # Write final citations.json
    OUT_PATH.write_text(
        json.dumps(all_citations, ensure_ascii=False, separators=(',', ':')))
    print(f"\nDone! {len(all_citations)} total citations extracted.")
    print(f"Written to {OUT_PATH}")

    # Quick stats
    from collections import Counter
    cats = Counter(c.get('category', 'unknown') for c in all_citations)
    works = Counter(c.get('work', '') for c in all_citations)
    print("\nBy category:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print("\nTop 20 works:")
    for work, count in works.most_common(20):
        print(f"  {work}: {count}")

if __name__ == '__main__':
    main()
