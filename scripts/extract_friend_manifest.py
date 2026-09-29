#!/usr/bin/env python3
import json
import sys
import os

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <result_json_path_or_url> <output_json_path>", file=sys.stderr)
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    if src.startswith("root://"):
        try:
            import fsspec
            with fsspec.open(src, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"fsspec open failed for {src}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        with open(src, "r") as f:
            data = json.load(f)

    merged = data.get("predictions")
    if merged is None:
        analysis = data.get("analysis")
        if analysis and isinstance(analysis, list) and len(analysis) > 0:
            merged = analysis[0].get("merged")
    if merged is None:
        print(f"Error: could not find 'predictions' or 'merged' structure in {src}. Type: {type(data)}, Keys/Len: {list(data.keys()) if isinstance(data, dict) else len(data)}", file=sys.stderr)
        sys.exit(1)

    with open(dst, "w") as out:
        json.dump({"FvT": merged}, out, indent=2)

    print(f"Successfully extracted friend manifest to {dst}")

if __name__ == "__main__":
    main()
