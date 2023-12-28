import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_xywh_to_cxcywh, box_xywh_to_xyxy


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "class_confidence", "track_id", "t"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        cropped_boxes = target['boxes'].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target

def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.shape[-2:]
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target

def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    return padded_image, target

def rotate(image, boxes, angle):
    """
        Rotate image and bounding box
        image: A Pil image (w, h)
        boxes: A tensors of dimensions (#objects, 4)

        Out: rotated image (w, h), rotated boxes
    """
    new_image = image.copy()
    new_boxes = boxes.clone()

    # Rotate image, expand = True
    w = image.width
    h = image.height
    cx = w / 2
    cy = h / 2
    new_image = new_image.rotate(angle, expand=True)
    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)
    # Get affine matrix
    AffineMatrix = torch.tensor([[alpha, beta, (1 - alpha) * cx - beta * cy],
                                 [-beta, alpha, beta * cx + (1 - alpha) * cy]])

    # Rotation boxes
    box_width = (boxes[:, 2] - boxes[:, 0]).reshape(-1, 1)
    box_height = (boxes[:, 3] - boxes[:, 1]).reshape(-1, 1)

    # Get corners for boxes
    x1 = boxes[:, 0].reshape(-1, 1)
    y1 = boxes[:, 1].reshape(-1, 1)

    x2 = x1 + box_width
    y2 = y1

    x3 = x1
    y3 = y1 + box_height

    x4 = boxes[:, 2].reshape(-1, 1)
    y4 = boxes[:, 3].reshape(-1, 1)

    corners = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)
    # corners.reshape(-1, 8)    #Tensors of dimensions (#objects, 8)
    corners = corners.reshape(-1, 2)  # Tensors of dimension (4* #objects, 2)
    corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim=1)  # (Tensors of dimension (4* #objects, 3))

    cos = np.abs(AffineMatrix[0, 0])
    sin = np.abs(AffineMatrix[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    AffineMatrix[0, 2] += (nW / 2) - cx
    AffineMatrix[1, 2] += (nH / 2) - cy

    # Apply affine transform
    rotate_corners = torch.mm(AffineMatrix, corners.t().to(torch.float64)).t()
    rotate_corners = rotate_corners.reshape(-1, 8)

    x_corners = rotate_corners[:, [0, 2, 4, 6]]
    y_corners = rotate_corners[:, [1, 3, 5, 7]]

    # Get (x_min, y_min, x_max, y_max)
    x_min, _ = torch.min(x_corners, dim=1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim=1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim=1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim=1)
    y_max = y_max.reshape(-1, 1)

    new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim=1)

    scale_x = new_image.width / w
    scale_y = new_image.height / h

    # Resize new image to (w, h)

    new_image = new_image.resize((w, h))

    # Resize boxes
    new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)
    return new_image, new_boxes

class Rotate:
    def __init__(self, angle=10) -> None:
        self.angle = angle

    def __call__(self, img, target):
        w, h = img.size
        whwh = torch.Tensor([w, h, w, h])
        boxes_xyxy = box_cxcywh_to_xyxy(target['boxes']) * whwh
        img, boxes_new = rotate(img, boxes_xyxy, self.angle)
        target['boxes'] = box_xyxy_to_cxcywh(boxes_new).to(boxes_xyxy.dtype) / (whwh + 1e-3)
        return img, target


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomCrop(object):
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        region = T.RandomCrop.get_params(img, (size, size))
        return crop(img, target, region)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self):
        self.num_of_max_pixel = 255.0

    def __call__(self, image, target=None):
        image = image / self.num_of_max_pixel
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            keys = []
            for k in target:
                if k != 'boxes':
                    keys.append(k)
            boxes = target["boxes"]

            boxes = box_xywh_to_xyxy(boxes)
            if not (boxes[:, 2:] > boxes[:, :2]).all():
                mask = (boxes[:, 2:] <= boxes[:, :2]).any(dim=-1)
                boxes = boxes[~mask, :]
                target['boxes'] = boxes
                for k in keys:
                    true_value = target[k][~mask]
                    target[k] = true_value

            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

            target['boxes'] = boxes
            if torch.isnan(boxes).any():
                nan_mask = torch.isnan(boxes).any(dim=-1)
                boxes = boxes[~nan_mask, :]
                target['boxes'] = boxes
                for k in keys:
                    true_value = target[k][~nan_mask]
                    target[k] = true_value

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
