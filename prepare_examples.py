"""Download attributed CC BY / public-domain teaching examples from Commons."""
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
ITEMS = [
    ("01_street_portrait", "Street fashion (27305297617).jpg"),
    ("02_street_style", "Street fashion (14029430420).jpg"),
    ("03_street_scene", "Street fashion (15758256296).jpg"),
    ("04_street_group", "02024 0843 Street fashion.jpg"),
    ("05_vintage_fashion", "Street fashion, Sydney 1945 SLNSW FL692463.jpg"),
]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "FashionCourseDemo/1.0 (educational attribution-preserving image download)"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def plain(value):
    return re.sub(r"<[^>]+>", "", value)


def main():
    folder = ROOT / "examples"
    folder.mkdir(exist_ok=True)
    manifest = []
    for stem, title in ITEMS:
        parameters = {"action": "query", "format": "json", "titles": "File:" + title, "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 960}
        data = json.loads(fetch("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(parameters)))
        info = next(iter(data["query"]["pages"].values()))["imageinfo"][0]
        metadata = info["extmetadata"]
        license_name = plain(metadata.get("LicenseShortName", {}).get("value", ""))
        if not any(term in license_name.lower() for term in ("cc by", "public domain", "pdm", "cc0")):
            raise RuntimeError(f"Review image license before redistribution: {title}, {license_name}")
        content = fetch(info.get("thumburl", info["url"]))
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
        image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        image.save(folder / f"{stem}.jpg", quality=92)
        manifest.append({"file": f"{stem}.jpg", "title": title, "source": info["descriptionurl"], "download_url": info.get("thumburl", info["url"]), "author": plain(metadata.get("Artist", {}).get("value", "")), "license": license_name, "license_url": metadata.get("LicenseUrl", {}).get("value", "https://creativecommons.org/publicdomain/mark/1.0/"), "changes": "Resized to at most 1200 pixels on the longest edge and re-encoded as JPEG. Result images additionally contain predicted boxes and person masks."})
        (folder / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Downloaded {stem}: {image.size}, {license_name}", flush=True)
    lines = ["# 示例图片来源与许可", "", "图片用于课程教学演示。以下许可适用于原图及相应的结果图，作者不代表认可本项目。图片已缩小并重新编码，结果图增加模型检测框和人物掩码。", ""]
    for entry in manifest:
        lines.extend([f"## {entry['file']}", f"- 原作品：[{entry['title']}]({entry['source']})", f"- 作者：{entry['author']}", f"- 许可：[{entry['license']}]({entry['license_url']})", ""])
    (folder / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
