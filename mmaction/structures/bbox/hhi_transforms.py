import numpy as np
import torch


def bbox2result_hhi(bboxes, p1_inds, p2_inds, labels, num_classes, thr=0.01):
    if bboxes.shape[0] == 0:
        return list(np.zeros((num_classes - 1, 0, 9), dtype=np.float32))

    p1_boxes = bboxes[p1_inds]
    p2_boxes = bboxes[p2_inds]

    p1_boxes = p1_boxes.cpu().numpy()
    p2_boxes = p2_boxes.cpu().numpy()
    scores = labels.cpu().numpy()  # rename for clarification
    assert p1_boxes.shape[0] == scores.shape[0]
    assert p1_boxes.shape[0] == p2_boxes.shape[0]

    # Although we can handle single-label classification, we still want scores
    assert scores.shape[-1] > 1

    # Robustly check for multi/single-label:
    if not hasattr(thr, '__len__'):
        multilabel = thr >= 0
        if thr < 0:
            thr = - thr
        thr = (thr, ) * num_classes
    else:
        multilabel = True

    # Check Shape
    assert scores.shape[1] == num_classes, f"scores: {scores.shape}, num_classes: {num_classes}"
    assert len(thr) == num_classes

    result = []
    for i in range(num_classes - 1):
        if multilabel:
            where = (scores[:, i + 1] > thr[i + 1])
        else:
            where = (scores[:, 1:].argmax(axis=1) == i) & (scores[:, i+1] > thr[i+1])
        result.append(
            np.concatenate((p1_boxes[where, :4], p2_boxes[where, :4], scores[where, i + 1:i + 2]),
                           axis=1))
    return result



def check_bboxes(bboxes):
    assert bboxes.shape[1] == 4
    x_diff = bboxes[:, 0] - bboxes[:, 2]
    invalid_x = (x_diff > 0).any()
    y_diff = bboxes[:, 1] - bboxes[:, 3]
    invalid_y = (y_diff > 0).any()
    valid = not (invalid_x or invalid_y)
    return valid


def bboxes2union(p1_bboxes, p2_bboxes):
    assert p1_bboxes.shape[0] == p2_bboxes.shape[0]
    check_p1 = check_bboxes(p1_bboxes)
    check_p2 = check_bboxes(p2_bboxes)
    assert check_p1 and check_p2
    
    union_x1, _ = torch.min(torch.stack([p1_bboxes[:, 0], p2_bboxes[:, 0]], 1), 1)
    union_y1, _ = torch.min(torch.stack([p1_bboxes[:, 1], p2_bboxes[:, 1]], 1), 1)
    union_x2, _ = torch.min(torch.stack([p1_bboxes[:, 2], p2_bboxes[:, 2]], 1), 1)
    union_y2, _ = torch.min(torch.stack([p1_bboxes[:, 3], p2_bboxes[:, 3]], 1), 1)
    union_bboxes = torch.stack([union_x1, union_y1, union_x2, union_y2], 1)
    
    return union_bboxes