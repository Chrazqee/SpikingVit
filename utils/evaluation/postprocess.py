from pprint import pprint

import torch

from utils.box_ops import box_cxcywh_to_xyxy


def postprocess(output, num_classes, conf_thre=0.7, img_hw=None):
    """
    :param img_hw: ...
    :param output: {'pred_logits' :tensor([B(31), 100, 4]), 'pred_boxes' :tensor([B(31), 100, 4])[cxcywh]}
    :param num_classes: 4
    :param conf_thre: ...
    :return: list(tensor(x1, y1, x2, y2, obj_conf, class_conf, class_pred), ...)
    """
    if img_hw is None:
        img_hw = [384, 640]
    pred_logits, pred_boxes = output['pred_logits'], output['pred_boxes']
    h, w = img_hw
    pred_boxes[:, :, 0::2] = pred_boxes[:, :, 0::2] * w
    pred_boxes[:, :, 1::2] = pred_boxes[:, :, 1::2] * h

    output = [None for _ in range(len(pred_logits))]
    for i, image_pred in enumerate(pred_logits):  # 1 帧 [100, 4]
        # 1、根据 `pred_logits` 将 class 为 3 的类别给 mask 掉，使用判断即可！
        image_pred_softmax = image_pred.softmax(-1)
        logit_values, logit_index = torch.topk(image_pred_softmax, k=1, dim=-1)
        class_conf, class_pred = logit_values, logit_index
        mask = class_pred == num_classes - 1
        mask = ~mask.squeeze(-1)  # [B, 100]

        bbox_pred = pred_boxes[i][mask]
        # convert cxcywh to xyxy
        bbox_pred = box_cxcywh_to_xyxy(bbox_pred)
        obj_conf = torch.ones(sum(mask), 1, device=bbox_pred.device)
        class_conf = class_conf[mask]
        class_pred = class_pred[mask]

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat([bbox_pred, obj_conf, class_conf, class_pred], dim=1)
        if len(conf_mask.shape) == 0:
            conf_mask = conf_mask.unsqueeze(-1)
        detections = detections[conf_mask]
        output[i] = detections

    return output


if __name__ == '__main__':
    output_x = {'pred_logits': torch.randn(4, 100, 4),
                'pred_boxes': torch.randn(4, 100, 4).sigmoid()}
    output_y = postprocess(output_x, 4, 0.7)
    pprint(output_y)
