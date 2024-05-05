import csv
import multiprocessing
import time
from collections import defaultdict

import numpy as np

from .hhi_evaluation import metrics, np_box_list, np_box_ops


def det2csv_hhi(results, custom_classes):
    csv_results = []
    for idx in range(len(results)):
        video_id = results[idx]['video_id']
        timestamp = results[idx]['timestamp']
        result = results[idx]['outputs']
        for label, _ in enumerate(result):
            for bbox in result[label]:
                bbox_ = tuple(bbox.tolist())
                if custom_classes is not None:
                    actual_label = custom_classes[label+1]
                else:
                    actual_label = label + 1
                csv_results.append((
                    video_id,
                    timestamp,
                ) + bbox_[:8] + (actual_label, ) + bbox_[8:])
    return csv_results


# results is organized by class
def results2csv_hhi(results, out_file, custom_classes=None):
    """Convert detection results to csv file."""
    csv_results = det2csv_hhi(results, custom_classes)

    # save space for float
    def to_str(item):
        if isinstance(item, float):
            return f'{item:.4f}'
        return str(item)

    with open(out_file, 'w') as f:
        for csv_result in csv_results:
            f.write(','.join(map(to_str, csv_result)))
            f.write('\n')


def print_time_hhi(message, start):
    """Print processing time."""
    print('==> %g seconds to %s' % (time.time() - start, message), flush=True)


def make_image_key_hhi(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return f'{video_id},{int(timestamp):04d}'


def read_csv_hhi(csv_file,
                 class_whitelist=None,
                 gt=False,
                 pred_max=-1):
    entries = defaultdict(list)
    p1_boxes = defaultdict(list)
    p2_boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)

    for row in reader:
        assert len(row) in [12, 13] , 'Wrong number of columns: ' + row
        if not gt:
            image_key = make_image_key_hhi(row[0], row[1])
        else:
            # TODO: modify annotation fps
            image_key = make_image_key_hhi(row[0], str(int(row[1])//5))
        x11, y11, x12, y12 = [float(n) for n in row[2:6]]
        x21, y21, x22, y22 = [float(n) for n in row[6:10]]
        action_id = int(row[10])
        if class_whitelist and action_id not in class_whitelist:
            continue

        score = 1.0
        if len(row) == 12: # prediction result, else it should be gt
            score = float(row[11])

        entries[image_key].append((score, action_id, y11, x11, y12, x12, y21, x21, y22, x22))
    
    for image_key in entries:
        entry = sorted(entries[image_key], key=lambda tup:-tup[0])
        if pred_max > 0 and pred_max < len(entry):
            entry = entry[:pred_max]
        p1_boxes[image_key] = [x[2:6] for x in entry]
        p2_boxes[image_key] = [x[6:] for x in entry]
        labels[image_key] = [x[1] for x in entry]
        scores[image_key] = [x[0] for x in entry]
    
    return p1_boxes, p2_boxes, labels, scores


def read_exclusions_hhi(exclusions_file):
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
    for row in reader:
        assert len(row) == 2, f'Expected only 2 columns, got: {row}'
        excluded.add(make_image_key_hhi(row[0], row[1]))
    return excluded


def read_labelmap_hhi(labelmap_file):
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids


def get_overlaps_and_scores_box_mode_hhi(detected_p1_boxes,
                                         detected_p2_boxes,
                                         detected_scores,
                                         groundtruth_p1_boxes,
                                         groundtruth_p2_boxes):

    detected_p1_boxlist = np_box_list.BoxList(detected_p1_boxes)
    detected_p1_boxlist.add_field('scores', detected_scores)
    detected_p2_boxlist = np_box_list.BoxList(detected_p2_boxes)
    gt_non_group_of_p1_boxlist = np_box_list.BoxList(groundtruth_p1_boxes)
    gt_non_group_of_p2_boxlist = np_box_list.BoxList(groundtruth_p2_boxes)

    iou_p1 = np_box_ops.iou(detected_p1_boxlist.get(), gt_non_group_of_p1_boxlist.get())
    iou_p2 = np_box_ops.iou(detected_p2_boxlist.get(), gt_non_group_of_p2_boxlist.get())
    scores = detected_p1_boxlist.get_field('scores')
    num_boxes = detected_p1_boxlist.num_boxes()
    return iou_p1, iou_p2, scores, num_boxes


def tpfp_single(tup, threshold=0.9):
    gt_p1_boxes, gt_p2_boxes, gt_labels, p1_boxes, p2_boxes, labels, scores = tup
    ret_scores, ret_tp_fp_labels = dict(), dict()
    all_labels = list(set(labels))

    for label in all_labels:
        # gt_inds = [i for i in range(len(gt_labels)) if gt_labels[i] == label]
        # det_inds = [i for i in range(len(labels)) if labels[i] == label]

        # gt_p1_box = []
        # gt_p2_box = []
        # for i in gt_inds:
        #     gt_p1_box.append(gt_p1_boxes[i])
        #     gt_p2_box.append(gt_p2_boxes[i])
        # gt_p1_box = np.array(gt_p1_box, dtype=np.float32).reshape(-1,4)
        # gt_p2_box = np.array(gt_p2_box, dtype=np.float32).reshape(-1,4)

        # p1_box = []
        # p2_box = []
        # score = []
        # for i in det_inds:
        #     p1_box.append(p1_boxes[i])
        #     p2_box.append(p2_boxes[i])
        #     score.append(scores[i])
        # p1_box = np.array(p1_box, dtype=np.float32).reshape(-1, 4)
        # p2_box = np.array(p2_box, dtype=np.float32).reshape(-1, 4)
        # score = np.array(score, dtype=np.float32).reshape(-1)

        gt_p1_box = np.array(
            [x for x, y in zip(gt_p1_boxes, gt_labels) if y == label]
        ).reshape(-1, 4)
        gt_p2_box = np.array(
            [x for x, y in zip(gt_p2_boxes, gt_labels) if y == label]
        ).reshape(-1, 4)
        p1_box = np.array(
            [x for x, y in zip(p1_boxes, labels) if y == label],
            dtype=np.float32
        ).reshape(-1,4)
        p2_box = np.array(
            [x for x, y in zip(p2_boxes, labels) if y == label],
            dtype=np.float32
        ).reshape(-1,4)
        score = np.array(
            [x for x, y in zip(scores, labels) if y == label],
            dtype=np.float32
        ).reshape(-1)
        # print(label, gt_p1_box.shape, p1_box.shape, score.shape, flush=True)
        assert len(p1_box) == len(score), f"p1_boxes: {p1_box.shape}, scores: {score.shape}"
        (iou_p1, iou_p2, _, num_boxes) = get_overlaps_and_scores_box_mode_hhi(
            detected_p1_boxes=p1_box,
            detected_p2_boxes=p2_box,
            detected_scores=score,
            groundtruth_p1_boxes=gt_p1_box,
            groundtruth_p2_boxes=gt_p2_box
        )
        iou = np.minimum(iou_p1, iou_p2)
        assert iou.shape == iou_p1.shape

        if gt_p1_box.size == 0:
            ret_scores[label] = score
            ret_tp_fp_labels[label] = np.zeros(num_boxes, dtype=bool)
            continue

        tp_fp_labels = np.zeros(num_boxes, dtype=bool)
        if iou.shape[1] > 0:
            is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
            for i in range(num_boxes):
                sorted_indices = np.argsort(iou[i])
                sorted_indices = sorted_indices[::-1]
                for gt_id in sorted_indices:
                    if iou_p1[i, gt_id] >= threshold and iou_p2[i, gt_id] >= threshold:
                        if not is_gt_box_detected[gt_id]:
                            tp_fp_labels[i] = True
                            is_gt_box_detected[gt_id] = True
                            break
        ret_scores[label], ret_tp_fp_labels[label] = score, tp_fp_labels
    
    return ret_scores, ret_tp_fp_labels


def hhi_eval(result_file,
             result_type,
             label_file,
             ann_file,
             exclude_file,
             verbose=True,
             ignore_empty_frames=True,
             custom_classes=None,
             pred_max=-1):
    
    start = time.time()
    categories, class_whitelist = read_labelmap_hhi(open(label_file))
    if custom_classes is not None:
        custom_classes = custom_classes[1:]
        assert set(custom_classes).issubset(set(class_whitelist))
        class_whitelist = custom_classes
        categories = [cat for cat in categories if cat['id'] in custom_classes]

    # loading gt, do not need gt score
    gt_p1_boxes, gt_p2_boxes, gt_labels, _ = read_csv_hhi(
        open(ann_file),
        class_whitelist,
        True,
        pred_max
    )
    if verbose:
        print_time_hhi('Reading GT results', start)

    if exclude_file is not None:
        excluded_keys = read_exclusions_hhi(open(exclude_file))
    else:
        excluded_keys = list()

    start = time.time()
    p1_boxes, p2_boxes, labels, scores = read_csv_hhi(
        open(result_file),
        class_whitelist,
        False,
        pred_max
    )
    if verbose:
        print_time_hhi('Reading Detection results', start)

    start = time.time()
    all_gt_labels = np.concatenate(list(gt_labels.values()))
    gt_count = {k: np.sum(all_gt_labels == k) for k in class_whitelist}

    pool = multiprocessing.Pool(32)
    if ignore_empty_frames:
        tups = [(gt_p1_boxes[k], gt_p2_boxes[k], gt_labels[k], p1_boxes[k], p2_boxes[k], labels[k], scores[k])
                for k in gt_p1_boxes if k not in excluded_keys]
    else:
        tups = [(gt_p1_boxes.get(k, np.zeros((0,4), dtype=np.float32)),
                 gt_p2_boxes.get(k, np.zeros((0,4), dtype=np.float32)),
                 gt_labels.get(k, []),
                 p1_boxes[k], p2_boxes[k], labels[k], scores[k])
                 for k in p1_boxes if k not in excluded_keys]
    rets = pool.map(tpfp_single, tups)
    # rets = []
    # threshold=0.5
    # if ignore_empty_frames:
    #     for k in gt_p1_boxes:
    #         if k in excluded_keys:
    #             continue
    #         tup = (gt_p1_boxes[k], gt_p2_boxes[k], gt_labels[k],
    #                p1_boxes[k], p2_boxes[k], labels[k], scores[k])
    #         rets.append(tpfp_single(tup, threshold=threshold))
    # else:
    #     for k in p1_boxes:
    #         if k in excluded_keys:
    #             continue
    #         tup = (gt_p1_boxes.get(k, np.zeros((0,4), dtype=np.float32)),
    #                gt_p2_boxes.get(k, np.zeros((0,4), dtype=np.float32)),
    #                gt_labels.get(k, []),
    #                p1_boxes[k], p2_boxes[k], labels[k], scores[k])
    #         rets.append(tpfp_single(tup, threshold=threshold))

    if verbose:
        print_time_hhi('Calculating TP/FP', start)

    start = time.time()
    scores, tpfps = defaultdict(list), defaultdict(list)
    for score, tpfp in rets:
        for k in score:
            scores[k].append(score[k])
            tpfps[k].append(tpfp[k])

    cls_AP = []
    total_tp = 0
    total_gt = 0
    for k in scores:
        scores[k] = np.concatenate(scores[k])
        tpfps[k] = np.concatenate(tpfps[k])
        precision, recall, tp, num_gt = metrics.compute_precision_recall(
            scores[k], tpfps[k], gt_count[k])
        if not tp is None:
            total_tp += tp
            total_gt += num_gt
        ap = metrics.compute_average_precision(precision, recall)
        class_name = [x['name'] for x in categories if x['id'] == k]
        assert len(class_name) == 1
        class_name = class_name[0]
        cls_AP.append((k, class_name, ap))
    overall = np.nanmean([x[2] for x in cls_AP])
    
    if verbose:
        print_time_hhi('Run Evaluator', start)
        print('Per-class results: ', flush=True)
        for k, class_name, ap in cls_AP:
            print(f'Class {class_name} AP: {ap:.4f}', flush=True)
        print('Overall Results: ', flush=True)
        print(f'Overall mAP: {overall:.4f}', flush=True)

    results = {}
    results['overall'] = overall
    results[f'recall@{pred_max}'] = total_tp/total_gt

    return results
    