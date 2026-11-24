"""Fetch and verify the pinned external comparison implementations."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parent
    records = json.loads((root / "upstream_sources.json").read_text(encoding="utf-8"))
    for row in records:
        target = root / "00_references" / row["local"]
        expected = row["upstream_sha256"]
        if target.exists() and sha256(target.read_bytes()).hexdigest() == expected:
            continue
        url = "https://raw.githubusercontent.com/{repo}/{commit}/{path}".format(**row)
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Source digest mismatch for {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"Verified {len(records)} upstream files.")


if __name__ == "__main__":
    main()
