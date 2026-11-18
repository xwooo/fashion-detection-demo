"""Download official weights in resumable ranges and verify SHA256 prefixes.

Requires curl (included in Windows 10/11). Torchvision also downloads weights
automatically; this optional helper is useful on slow connections.
"""
import hashlib
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FOLDER = Path(__file__).resolve().parent / ".cache" / "torch" / "hub" / "checkpoints"
FILES = ["fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", "deeplabv3_resnet50_coco-cd0a2569.pth"]
CHUNK = 8 * 1024 * 1024


def verified(path, expected):
    if not path.exists():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().startswith(expected)


def main():
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for this optional helper")
    FOLDER.mkdir(parents=True, exist_ok=True)
    jobs, models = [], []
    for name in FILES:
        expected = name.rsplit("-", 1)[1].split(".")[0]
        target = FOLDER / name
        if verified(target, expected):
            print(f"Verified cache: {name}", flush=True)
            continue
        url = "https://download.pytorch.org/models/" + name
        headers = subprocess.check_output([curl, "--head", "--silent", "--show-error", "--fail", "--max-time", "45", url], text=True)
        size = int(re.findall(r"(?im)^content-length:\s*(\d+)", headers)[-1])
        parts = []
        for start in range(0, size, CHUNK):
            end = min(start + CHUNK, size) - 1
            path = FOLDER / f"{name}.chunk-{start:09d}"
            parts.append(path)
            jobs.append((url, start, end, path))
        models.append((target, expected, parts))

    def fetch(job):
        url, start, end, path = job
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        subprocess.run([curl, "--silent", "--show-error", "--fail", "--location", "--retry", "5", "--retry-all-errors", "--max-time", "180", "--range", f"{start}-{end}", "--output", str(path), url], check=True)
        if path.stat().st_size != end - start + 1:
            raise RuntimeError(f"Wrong range size: {path.name}")
        print(f"Downloaded {path.name}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, jobs))
    for target, expected, parts in models:
        partial = target.with_suffix(".verified-part")
        with partial.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output)
        if not verified(partial, expected):
            raise RuntimeError(f"SHA256 mismatch for {target.name}")
        partial.replace(target)
        print(f"Verified model: {target.name}", flush=True)


if __name__ == "__main__":
    main()
