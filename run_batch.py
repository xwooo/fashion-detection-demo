"""Reproduce experiment outputs with one model load."""
import argparse
import json
from pathlib import Path
from fashion_demo.pipeline import FashionPipeline, ROOT, read_image


def main():
    parser = argparse.ArgumentParser(description="批量生成检测与人物分割结果")
    parser.add_argument("--input", type=Path, default=ROOT / "examples")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    paths = [args.input] if args.input.is_file() else sorted(p for p in args.input.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        parser.error("未找到输入图片。")
    pipeline = FashionPipeline(args.device)
    summary, errors = [], []
    for path in paths:
        try:
            result = pipeline.run(read_image(path), args.threshold, args.alpha)
            result.metadata["source_file"] = path.name
            result.save(args.output / path.stem)
            summary.append(result.metadata)
            print(f"{path.name}: {len(result.metadata['detections'])} targets, {result.metadata['inference_seconds']:.2f}s", flush=True)
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
            print(f"ERROR {path.name}: {exc}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"results": summary, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
