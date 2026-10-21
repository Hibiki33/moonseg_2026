# Segmentation metrics

The evaluator sums an integer pixel confusion matrix before computing metrics, both across images and across distributed ranks. Rows are ground-truth classes; columns are predicted classes. Padding and other targets outside IDs 0–4 are ignored. Sky (ID 4) remains a valid class.

For class c, support is the ground-truth pixel count TP + FN. Only classes with positive support enter the macro averages. This matches the challenge's [ground-truth-present class set](https://insight.onescience.cn/competition/moonseg?event_id=c2f557a58ff2e9f4265d56f112a7e01&tab=info&section=validation).

| Output | Definition |
| --- | --- |
| `class_IoU` | TP / (TP + FP + FN), or `null` for absent ground-truth classes |
| `mIoU` | Arithmetic mean of `class_IoU` over ground-truth-present classes |
| `class_accuracy` | TP / (TP + FN), or `null` for absent ground-truth classes |
| `class_recall` | Identical to `class_accuracy`; recall names its denominator explicitly |
| `mAcc` | Arithmetic mean of per-class accuracy over ground-truth-present classes |
| `class_support` | Ground-truth pixel counts, including zero for absent classes |
| `pixel_accuracy` | Sum of TP / total valid ground-truth pixels |
| `confusion_matrix` | Accumulated integer counts in class order 0–4 |

All accuracy and IoU values are fractions in [0,1]. If every target pixel is ignored, `mIoU`, `mAcc`, `pixel_accuracy`, and each per-class IoU/accuracy/recall are `null`; all supports and confusion counts are zero. A prediction of a class absent from ground truth does not add that class to the averaging set, but it still counts as a false negative for the true class.

The top-level result pools all evaluated pixels. The `scenes` mapping computes the same metrics separately from each scene's summed confusion matrix. Scene 5 is the local validation split; its result does not represent a leaderboard score. The official weighted total additionally requires the organizer's frozen scene weights; no numerical weights are assumed by this evaluator.

For example, with confusion rows `[3,1,0,0,0]` and `[1,1,0,0,1]` for ground-truth classes 0 and 1, and all other rows zero, supports are 4 and 3. The class accuracies are 3/4 and 1/3, so `mAcc = 13/24`; `mIoU = (3/5 + 1/4)/2`, while `pixel_accuracy = 4/7`. These statistics answer different questions and are reported separately.

The mAcc denominator also follows the explicit intersection/ground-truth-area implementation in [MMSegmentation IoUMetric](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/mmseg/evaluation/metrics/iou_metric.py). mAcc is an additional diagnostic; the challenge's main score remains its scene-weighted mIoU.
