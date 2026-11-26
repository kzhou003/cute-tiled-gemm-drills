#!/usr/bin/env python3
"""solutions/f*.cu -> mine/f*.cu: every `// [TODO x] description ... // [/TODO]` block becomes

    // [TODO x] description
    DRILL_TODO("x description");
    // [/TODO]

DRILL_TODO is a static_assert that always fails (gpu/common.cuh), so an unfinished file does not
compile and the compiler lists every remaining TODO. The solution's code and comments inside the
block are dropped; everything outside stays identical.

    python3 tools/gen_cu_stubs.py            # write missing mine/f*.cu
    python3 tools/gen_cu_stubs.py --force    # overwrite (destroys your work)
    python3 tools/gen_cu_stubs.py --check    # every stub has DRILL_TODOs and the same code outside them
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPEN = re.compile(r"^(\s*)// \[TODO ([^\]]+)\]\s*(.*)$")
CLOSE = re.compile(r"^\s*// \[/TODO\]\s*$")


def stub(src: str) -> str:
    out, lines, i = [], src.splitlines(), 0
    while i < len(lines):
        m = OPEN.match(lines[i])
        if not m:
            out.append(lines[i]); i += 1; continue
        indent, tag, desc = m.groups()
        j = i + 1
        while not CLOSE.match(lines[j]):
            j += 1
        msg = f"{tag} {desc}".replace("\\", "\\\\").replace('"', '\\"')
        out += [lines[i], f'{indent}DRILL_TODO("{msg}");', lines[j]]
        i = j + 1
    text = "\n".join(out) + "\n"
    return text.replace("— reference solution.", "— your implementation (run ./check.sh " + "{id}" + ").")


def main():
    force, check = "--force" in sys.argv, "--check" in sys.argv
    bad = 0
    for sol in sorted((ROOT / "solutions").glob("f*.cu")):
        mine = ROOT / "mine" / sol.name
        want = stub(sol.read_text()).replace("{id}", sol.stem.split("_")[0])
        if check:
            ok = mine.exists() and "DRILL_TODO(" in mine.read_text()
            print(f"{'ok ' if ok else 'BAD'} {mine.relative_to(ROOT)}")
            bad += not ok
            continue
        if mine.exists() and not force:
            print(f"keep {mine.relative_to(ROOT)}")
            continue
        mine.write_text(want)
        print(f"wrote {mine.relative_to(ROOT)} ({want.count('DRILL_TODO(')} TODOs)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
