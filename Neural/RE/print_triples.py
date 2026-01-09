import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--with_conf", action="store_true")
    args = ap.parse_args()

    path = Path(args.jsonl)
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        conf = float(obj.get("conf", 1.0))
        if conf < args.min_conf:
            continue

        e1 = obj["head"]
        r = obj["relation"]
        e2 = obj["tail"]

        if args.with_conf:
            print(f"({e1}, {r}, {e2})  conf={conf:.4f}")
        else:
            print(f"({e1}, {r}, {e2})")

        n += 1
        if n >= args.max:
            break

if __name__ == "__main__":
    main()
