"""Inference for Ultralytics raw YOLO detection exports (NCHW, batch 1)."""
import ast
import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps


def nms(boxes, scores, classes, threshold, limit=300):
    """Class-aware NMS; boxes are xyxy in model input coordinates."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < limit:
        index = order[0]
        keep.append(int(index))
        other = order[1:]
        low = np.maximum(boxes[index, :2], boxes[other, :2])
        high = np.minimum(boxes[index, 2:], boxes[other, 2:])
        intersection = np.maximum(high - low, 0).prod(axis=1)
        area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
        overlap = intersection / np.maximum(area[index] + area[other] - intersection, 1e-7)
        order = other[(overlap <= threshold) | (classes[other] != classes[index])]
    return keep


class Detector:
    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("仅支持单输入、单输出的 YOLO 检测模型")
        self.input = inputs[0]
        shape = self.input.shape
        if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(v, int) for v in shape[2:]):
            raise ValueError(f"需要固定尺寸 [1,3,H,W] 输入，实际为 {shape}")
        if self.input.type != "tensor(float)":
            raise ValueError("需要 float32 模型")
        self.height, self.width = shape[2:]
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("task") != "detect":
            raise ValueError("模型元数据 task 必须为 detect")
        names = ast.literal_eval(metadata["names"])
        self.names = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        if sorted(self.names) != list(range(len(self.names))):
            raise ValueError("类别索引必须从 0 连续编号")
        output = outputs[0].shape
        if len(output) != 3 or output[0] != 1 or output[1] != 4 + len(self.names):
            raise ValueError(f"需要原始 YOLO 输出 [1,4+类别数,N]，实际为 {output}")
        self.info = {"model": path.name, "input_size": [self.width, self.height],
                     "classes": self.names, "provider": self.session.get_providers()[0]}

    def predict(self, image, confidence, iou):
        started = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        gain = min(self.width / width, self.height / height)
        resized = (round(width * gain), round(height * gain))
        left = round((self.width - resized[0]) / 2 - 0.1)
        top = round((self.height - resized[1]) / 2 - 0.1)
        padded = Image.new("RGB", (self.width, self.height), (114, 114, 114))
        padded.paste(image.resize(resized, Image.Resampling.BILINEAR), (left, top))
        tensor = np.ascontiguousarray(np.asarray(padded).transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        infer_start = time.perf_counter()
        prediction = self.session.run(None, {self.input.name: tensor})[0][0].T
        inference_ms = (time.perf_counter() - infer_start) * 1000
        classes = prediction[:, 4:].argmax(axis=1)
        scores = prediction[np.arange(len(prediction)), classes + 4]
        sicktree_ids = [index for index, name in self.names.items() if name == "sicktree"]
        valid = (scores >= confidence) & np.isfinite(prediction).all(axis=1) & np.isin(classes, sicktree_ids)
        prediction, scores, classes = prediction[valid], scores[valid], classes[valid]
        boxes = np.concatenate((prediction[:, :2] - prediction[:, 2:4] / 2,
                                prediction[:, :2] + prediction[:, 2:4] / 2), axis=1)
        keep = nms(boxes, scores, classes, iou)
        boxes = boxes[keep]
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / gain, 0, width)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / gain, 0, height)
        detections = []
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max(13, round(max(width, height) / 65)))
        line = max(2, round(max(width, height) / 400))
        for index, box in zip(keep, boxes):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            class_id = int(classes[index])
            name = self.names[class_id]
            detections.append({"class_id": class_id, "class_name": name,
                               "confidence": round(float(scores[index]), 5),
                               "box": [round(float(v), 2) for v in box]})
            color = "#e86d45" if name == "sicktree" else "#20b78c"
            xy = tuple(float(v) for v in box)
            draw.rectangle(xy, outline=color, width=line)
            label = f"{name} {scores[index]:.0%}"
            bounds = draw.textbbox((0, 0), label, font=font)
            label_height = bounds[3] - bounds[1] + 10
            y = max(0, xy[1] - label_height)
            draw.rectangle((xy[0], y, min(width, xy[0] + bounds[2] + 10), y + label_height), fill=color)
            draw.text((xy[0] + 5, y + 4 - bounds[1]), label, fill="white", font=font)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return {"detections": detections, "image_size": [width, height],
                "inference_ms": round(inference_ms, 1),
                "total_ms": round((time.perf_counter() - started) * 1000, 1)}, buffer.getvalue()
