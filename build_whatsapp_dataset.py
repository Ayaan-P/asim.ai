#!/usr/bin/env python3
"""
Convert a raw WhatsApp export to JSONL that looks like

    Ayaan: Yo send the link <|response|> 100%

Usage examples
--------------
# Fast sliding window
python build_whatsapp_dataset.py chat.txt asim.jsonl --sender "Asim" \
       --ctx_mode window --max_ctx 6

# Semantic retrieval (embed + cosine)
python build_whatsapp_dataset.py chat.txt asim.jsonl --sender "Asim" \
       --ctx_mode retrieve --top_k 12 --select_k 6
"""
import argparse, json, re, html, itertools, textwrap
from pathlib import Path
from typing import List, Tuple, Iterable, Iterator

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from tqdm import tqdm
except ImportError:                       
    def tqdm(x, *a, **k): return x

NBSP  = "\u00A0"
NNBSP = "\u202F"
LTR   = "\u200E?"

HEADER = re.compile(
    rf"^\[{LTR}"
    rf"(\d{{1,2}}/\d{{1,2}}/\d{{2,4}})[ {NBSP}]?,{LTR}\s*"          # date
    rf"(\d{{1,2}}:\d{{2}}(?:\:\d{{2}})?)[ {NBSP}{NNBSP}]?"          # time
    rf"(?:AM|PM)?\]{LTR}\s*"
    rf"([^:]+):\s*"                                                 # sender
)

BAN_LINE = re.compile(
    r"(Messages and calls are end[-‑]to[-‑]end|Voice call|Video call|"
    r"Call (?:failed|ended)|added|joined|left using|You created group|"
    r"missed voice|missed video|\+\d{1,3}\s*\(\d{3}\)\s*\d{3}[-‑]\d{4})",
    re.I,
)
JUNK_INNER = re.compile(
    r"(‎(?:image|sticker)\s+omitted|<This message was deleted>|https?://\S+)",
    re.I,
)
TS_INLINE = re.compile(r"\[\d{1,2}/\d{1,2}/\d{2,4},.*?\]")

def _parse(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """Yield (sender, body) tuples preserving multi‑line messages."""
    buf, sender = [], None
    for line in itertools.chain(lines, ["[99/99/9999, 99:99] sentinel:"]):
        m = HEADER.match(line)
        if m:
            if buf and sender is not None:
                yield sender, "\n".join(buf).strip()
            sender = m.group(3).strip()
            buf = [line[m.end():]]
        else:
            buf.append(line.rstrip("\n"))

def _clean(txt: str) -> str:
    if BAN_LINE.search(txt):
        return ""
    txt = html.unescape(txt)
    txt = JUNK_INNER.sub("", txt)
    txt = re.sub(r"^(Asim|Ayaan Pupala|Crispy|Sam|Paul):\s*", "", txt)
    txt = TS_INLINE.sub("", txt)
    return re.sub(r"\s+", " ", txt).strip()


class WindowSel:
    def __init__(self, n: int = 6): self.n = n
    def select(self, buf: List[str]) -> List[str]:
        return buf[-self.n:]

class RetrieveSel:
    def __init__(self, sender: str, top_k: int = 12, sel_k: int = 6,
                 model: str = "all-mpnet-base-v2"):
        if SentenceTransformer is None:
            raise ImportError("pip install sentence-transformers")
        self.sender_tag = f"{sender}:"
        self.top_k, self.sel_k = top_k, sel_k
        self.emb = SentenceTransformer(model)

    def select(self, buf: List[str]) -> List[str]:
        if not buf:
            return []
        chunk = buf[-self.top_k:]
        # anchor = latest *other‑speaker* line, fallback last line
        anchor_txt = next((l for l in reversed(chunk)
                           if not l.startswith(self.sender_tag)), chunk[-1])
        anchor = self.emb.encode(anchor_txt, normalize_embeddings=True)
        cand   = self.emb.encode(chunk, normalize_embeddings=True)
        scores = (anchor @ cand.T).tolist()
        idx = sorted(range(len(scores)),
                     key=scores.__getitem__, reverse=True)[:self.sel_k]
        idx.sort()
        return [chunk[i] for i in idx]

def build(raw: str, sender: str, sel, out: Path) -> int:
    ctx_buf: List[str] = []
    rows = 0
    with out.open("w", encoding="utf-8") as fout:
        for who, msg in tqdm(_parse(raw.splitlines()), unit="msg"):
            msg = _clean(msg)
            if not msg:
                continue

            if who == sender:
                # build context **before** pushing this new reply
                ctx_lines = list(dict.fromkeys(sel.select(ctx_buf)))
                context = " ".join(ctx_lines)
                if context:                                  # avoid empty ctx rows
                    json.dump({"text": f"{context} <|response|> {msg}"},
                              fout, ensure_ascii=False)
                    fout.write("\n")
                    rows += 1
                # now add this reply so it can appear later, but not in same row
                ctx_buf.append(f"{who}: {msg}")
            else:
                ctx_buf.append(f"{who}: {msg}")
    return rows

def main():
    pa = argparse.ArgumentParser(description=textwrap.dedent(__doc__))
    pa.add_argument("infile"); pa.add_argument("outfile")
    pa.add_argument("--sender", default="Asim")
    pa.add_argument("--ctx_mode", choices=["window", "retrieve"], default="window")
    pa.add_argument("--max_ctx", type=int, default=6)
    pa.add_argument("--top_k", type=int, default=12)
    pa.add_argument("--select_k", type=int, default=6)
    args = pa.parse_args()

    selector = WindowSel(args.max_ctx) if args.ctx_mode == "window" \
        else RetrieveSel(args.sender, args.top_k, args.select_k)

    raw = Path(args.infile).read_text(encoding="utf-8", errors="ignore")
    n = build(raw, args.sender, selector, Path(args.outfile))
    print(f"✔ wrote {n:,} rows → {args.outfile}")

if __name__ == "__main__":
    main()
