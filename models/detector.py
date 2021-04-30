import math, os
from numpy.core.shape_base import stack
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from models.yolo_utils import coco_names, coco_colors, non_max_suppression
from PIL import Image, ImageDraw
from models.base import BaseModel


class YOLOv5(BaseModel):
    def __init__(
        self, model_type="yolov5x", long_side=640, thr=0.25, device="cpu", fp16=False
    ):
        self.model_type = model_type
        self.device = torch.device(device)
        home_dir = os.environ.get("SVAS_HOME", "/mnt/svas")
        torch.hub.set_dir(os.path.join(home_dir, "data/models/tmp"))
        self.model = torch.hub.load(
            "xchani/yolov5", self.model_type, autoshape=False, pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.fp16 = fp16

        if self.fp16:
            self.model.half()

        self.long_side = long_side
        self.stride = int(self.model.stride.max())
        self.conf = thr
        self.iou = 0.45
        self.classes = None

    def load_images(self, image_path):
        if isinstance(image_path, str):
            image_path = [image_path]
        images = []
        for path in image_path:
            img = Image.open(path)
            img = np.array(img)

            img = torch.from_numpy(np.ascontiguousarray(img)).div(255)
            images.append(img)

        images = torch.stack(images)
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()
        images = images.to(self.device)

        return images

    def save_images(self, image_path, result):
        if isinstance(image_path, str):
            image_path = [image_path]
        colors = coco_colors()
        num_colors = len(colors)

        for i, r in enumerate(result):
            img = Image.open(image_path[i])
            save_path = image_path[i].replace(".jpg", "_plot.jpg")
            for obj in r:
                bbox = obj[:4].cpu().numpy()
                conf = float(obj[-2])
                cls = int(obj[-1])
                cls_name = coco_names[cls]

                ImageDraw.Draw(img).rectangle(
                    bbox, width=4, outline=colors[cls % num_colors]
                )
                img.save(save_path)

    def print(self, result):
        for i, r in enumerate(result):
            print(f"#Image {i}:")
            for obj in r:
                bbox = obj[:4].cpu().numpy()
                conf = float(obj[-2])
                cls = coco_names[int(obj[-1])]
                print(bbox, conf, cls)

    def benchmark(self, image_path, batch_size=8, trial=1000):
        if isinstance(image_path, str):
            image_path = [image_path] * batch_size
        assert len(image_path) == batch_size
        # regardless of I/O bottleneck
        images = self.load_images(image_path)
        images, _, _, _ = self.resize(images)
        for i in tqdm(range(trial)):
            result = self.infer(images)
    
    def benchmark_concate(self, image_path, batch_size=8, trial=1000, group_size=4):
        assert int(math.sqrt(group_size)) == math.sqrt(group_size)
        if isinstance(image_path, str):
            image_path = [image_path] * batch_size
        assert len(image_path) == batch_size
        # regardless of I/O bottleneck
        images = self.load_images(image_path)
        print("Shape1", images.shape)

        ratio = int(math.sqrt(group_size))
        old_long_side = self.long_side
        self.long_side = int(self.long_side / ratio)
        images, _, _, _ = self.resize(images)

        n, c, h, w = images.shape
        print("Shape2", images.shape, h,w )

        images = torch.split(images, group_size, 0)
        [print("shape-grpu:", group.shape) for group in images]

        images = [torch.reshape(group, (c, ratio*h, ratio*w)) for group in images]
        images = torch.stack(images)
        print("Shape3", images.shape)
        self.long_side = old_long_side

        images, _, _, _ = self.resize(images)

        print("Shape4", images.shape)
        for i in tqdm(range(trial)):
            result = self.infer(images)

    def make_divisible(self, x):
        return math.ceil(x / self.stride) * self.stride

    def resize(self, images, fill_color=114):
        n, c, h, w = images.shape

        # scale
        if h > w:
            unpad_h, unpad_w = self.long_side, round(self.long_side / h * w)
            ratio = self.long_side / h
        else:
            unpad_h, unpad_w = round(self.long_side / w * h), self.long_side
            ratio = self.long_side / w
        if (h, w) != (unpad_h, unpad_w):
            images = F.interpolate(
                images, size=(unpad_h, unpad_w), mode="bilinear", align_corners=False
            )

        # make it divisible
        pad_h, pad_w = self.make_divisible(unpad_h), self.make_divisible(unpad_w)

        # pad to make it divisible by stride
        if (h, w) != (pad_h, pad_w):
            top = (pad_h - unpad_h) // 2
            left = (pad_w - unpad_w) // 2
            images_pad = torch.full(
                (n, c, pad_h, pad_w), fill_color / 255, device=self.device
            )
            images_pad[:, :, top : top + unpad_h, left : left + unpad_w] = images
        else:
            top = left = 0
            images_pad = images

        if self.fp16:
            images_pad = images_pad.half()

        return images_pad, (h, w), ratio, (top, left)

    def scale_coords(self, coords, size, ratio, pad):
        # each image has different number of objects, thus can't stack together
        n = len(coords)
        for i in range(n):
            coords[i][:, [0, 2]] -= pad[1]
            coords[i][:, [1, 3]] -= pad[0]
            coords[i][:, :4] /= ratio

            coords[i][:, 0].clamp_(0, size[1])
            coords[i][:, 1].clamp_(0, size[0])
            coords[i][:, 2].clamp_(0, size[1])
            coords[i][:, 3].clamp_(0, size[0])

    def infer(self, images):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = images.shape[0]
        images, size, ratio, pad = self.resize(images)
        with torch.no_grad():
            result, _ = self.model(images)
            result = non_max_suppression(result, self.conf, self.iou, self.classes)
            self.scale_coords(result, size, ratio, pad)
        # result = [r.cpu().numpy() for r in result]
        return result

        # ret = []
        # result = [r.cpu().numpy() for r in result]
        # for i in range(len(result)):
        #     cur = []
        #     for j in range(len(result[i])):
        #         item = {
        #             "x1": result[i][j][0] / size[0],
        #             "y1": result[i][j][1] / size[0],
        #             "x2": result[i][j][2] / size[1],
        #             "y2": result[i][j][3] / size[1],
        #             "label": result[i][j][5],
        #             "prob": result[i][j][4],
        #         }
        #         cur.append(item)
        #     ret.append(cur)
        # return ret


if __name__ == "__main__":
    import os

    home_dir = os.getenv("SVAS_HOME", "/mnt/svas")
    image_path = os.path.join(home_dir, "demo/demo.jpg")

    det = YOLOv5(model_type="yolov5x", long_side=160, device="cuda:0", fp16=False)
    det.benchmark(image_path, batch_size=1)

    det = YOLOv5(model_type="yolov5x", long_side=640, device="cuda:0", fp16=False)
    det.benchmark_concate(image_path, batch_size=1, group_size=16)
