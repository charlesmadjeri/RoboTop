import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO, settings

SIZE = 320


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--base", default="yolov8n.pt")
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    settings.update({"sync": False})
    model = YOLO(args.base)
    model.train(data=str(args.data), imgsz=SIZE, epochs=args.epochs, batch=32, device="cpu", plots=False,
                project=str(args.runs.resolve()), name="train", exist_ok=True)
    exported = Path(model.export(format="onnx", imgsz=SIZE, opset=12, simplify=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, args.out)
    gpu = Path(model.export(format="ncnn", imgsz=SIZE))
    shutil.copytree(gpu, args.out.with_name(f"{args.out.stem}_ncnn"), dirs_exist_ok=True)


if __name__ == "__main__":
    main()
