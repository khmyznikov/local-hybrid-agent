import hashlib
import os
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja"
EXPECTED_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
OUTPUT = Path(
    os.getenv(
        "QWEN38_REFERENCE_TEMPLATE",
        "/home/gkhmyznikov/models/Qwen3.8-27B-reference/chat_template.jinja",
    )
)


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"First-party template hash changed: expected {EXPECTED_SHA256}, got {digest}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(OUTPUT)
    print(f"template={OUTPUT}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()