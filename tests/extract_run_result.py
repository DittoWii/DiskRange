#!/usr/bin/env python3
import re
import sys

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
RUN_RESULT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")


def main() -> int:
    values = {}
    for path in sys.argv[1:]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = ANSI_ESCAPE_RE.sub("", raw_line.rstrip("\n"))
                match = RUN_RESULT_RE.search(line)
                if match is None:
                    continue
                key, value = match.groups()
                values[key] = value

    for key, value in values.items():
        print(f"{key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
