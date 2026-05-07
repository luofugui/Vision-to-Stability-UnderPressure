import argparse
import gc
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_signals(pkl_path):
    """Extract ground-truth and predicted contact signals from a pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    try:
        if "targets" in data and "predictions" in data:
            gt_raw = np.asarray(data["targets"]["contact"])
            pred_raw = np.asarray(data["predictions"]["contact"])

            print(f"\n{os.path.basename(pkl_path)}")
            print(f"  GT contact shape:   {gt_raw.shape}")
            print(f"  Pred contact shape: {pred_raw.shape}")

            gt = gt_raw
            pred = pred_raw

        else:
            print(f"Available keys in pkl: {data.keys()}")
            raise KeyError("Cannot find contact keys in pkl.")
        return gt, pred
    except Exception as e:
        print(f"Error extracting signals: {e}")
        return None, None


def get_segments(binary_signal):
    """
    Extract contact segments from a binary signal.

    Returns half-open intervals: [(start_idx, end_idx), ...].
    Segment length in frames is end_idx - start_idx.
    """
    binary_signal = np.asarray(binary_signal).astype(int)
    padded = np.concatenate(([0], binary_signal, [0]))
    diff = np.diff(padded)

    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    return list(zip(onsets, offsets))


def segment_iou(gt_segment, pred_segment):
    """Calculate IoU between two half-open contact segments."""
    g_s, g_e = gt_segment
    p_s, p_e = pred_segment

    intersection = max(0, min(g_e, p_e) - max(g_s, p_s))
    if intersection == 0:
        return 0.0

    union = (g_e - g_s) + (p_e - p_s) - intersection
    return intersection / union if union > 0 else 0.0


def match_segments_iou(gt_segments, pred_segments, min_iou=0.0):
    """
    One-to-one match GT and predicted contact segments by descending IoU.

    This does not bridge gaps. Fragmented predictions stay as separate segments:
    one fragment may match a GT segment, while the remaining fragments are counted
    as unmatched predictions.
    """
    matched_pairs = []
    unmatched_gt = set(range(len(gt_segments)))
    unmatched_pred = set(range(len(pred_segments)))

    if not gt_segments or not pred_segments:
        return matched_pairs, sorted(unmatched_gt), sorted(unmatched_pred)

    iou_records = []
    for gt_idx, gt_segment in enumerate(gt_segments):
        g_s, g_e = gt_segment
        for pred_idx, pred_segment in enumerate(pred_segments):
            p_s, p_e = pred_segment

            if p_e <= g_s:
                continue
            if p_s >= g_e:
                break

            iou = segment_iou(gt_segment, pred_segment)
            if iou > min_iou:
                iou_records.append((iou, gt_idx, pred_idx))

    iou_records.sort(key=lambda x: x[0], reverse=True)

    for iou, gt_idx, pred_idx in iou_records:
        if gt_idx in unmatched_gt and pred_idx in unmatched_pred:
            matched_pairs.append((gt_idx, pred_idx, iou))
            unmatched_gt.remove(gt_idx)
            unmatched_pred.remove(pred_idx)

    return matched_pairs, sorted(unmatched_gt), sorted(unmatched_pred)


def calculate_contact_time_errors(
    gt_segments,
    pred_segments,
    matched_pairs,
    unmatched_gt,
    unmatched_pred,
    ms_per_frame,
):
    matched_errors_ms = []
    missed_gt_errors_ms = []
    false_pred_errors_ms = []

    for match in matched_pairs:
        gt_idx, pred_idx = match[:2]
        g_s, g_e = gt_segments[gt_idx]
        p_s, p_e = pred_segments[pred_idx]

        gt_duration = g_e - g_s
        pred_duration = p_e - p_s
        matched_errors_ms.append(abs(pred_duration - gt_duration) * ms_per_frame)

    for gt_idx in unmatched_gt:
        g_s, g_e = gt_segments[gt_idx]
        missed_gt_errors_ms.append((g_e - g_s) * ms_per_frame)

    for pred_idx in unmatched_pred:
        p_s, p_e = pred_segments[pred_idx]
        false_pred_errors_ms.append((p_e - p_s) * ms_per_frame)

    return matched_errors_ms, missed_gt_errors_ms, false_pred_errors_ms


def overlap_length(a, b):
    a_s, a_e = a
    b_s, b_e = b
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def subtract_intervals(interval, blockers):
    pieces = [interval]

    for b_s, b_e in blockers:
        new_pieces = []
        for p_s, p_e in pieces:
            if b_e <= p_s or b_s >= p_e:
                new_pieces.append((p_s, p_e))
            else:
                if p_s < b_s:
                    new_pieces.append((p_s, b_s))
                if b_e < p_e:
                    new_pieces.append((b_e, p_e))
        pieces = new_pieces

        if not pieces:
            break

    return pieces


def evaluate_one_contact_channel(gt_probs, pred_probs, fps=50, threshold=0.5, min_iou=0.0):
    """Coverage-based contact-time MAE for one continuous contact channel."""
    ms_per_frame = 1000.0 / fps

    gt_binary = (np.asarray(gt_probs) >= 0.5).astype(int)
    pred_binary = (np.asarray(pred_probs) > threshold).astype(int)

    gt_segments = get_segments(gt_binary)
    pred_segments = get_segments(pred_binary)

    covered_gt_errors_ms = []
    missed_gt_errors_ms = []

    for gt_seg in gt_segments:
        gt_duration = gt_seg[1] - gt_seg[0]

        pred_overlap_duration = sum(
            overlap_length(gt_seg, pred_seg)
            for pred_seg in pred_segments
        )

        error_ms = abs(gt_duration - pred_overlap_duration) * ms_per_frame

        if pred_overlap_duration > 0:
            covered_gt_errors_ms.append(error_ms)
        else:
            missed_gt_errors_ms.append(error_ms)

    false_pred_errors_ms = []

    for pred_seg in pred_segments:
        overlapping_gt_segments = [
            gt_seg for gt_seg in gt_segments
            if overlap_length(pred_seg, gt_seg) > 0
        ]

        outside_pieces = subtract_intervals(pred_seg, overlapping_gt_segments)

        for piece_s, piece_e in outside_pieces:
            false_pred_errors_ms.append((piece_e - piece_s) * ms_per_frame)

    errors_ms = covered_gt_errors_ms + missed_gt_errors_ms + false_pred_errors_ms

    error_count = len(errors_ms)
    total_error_ms = float(np.sum(errors_ms)) if error_count else 0.0
    ct_mae = total_error_ms / error_count if error_count else 0.0

    return {
        "ct_mae": ct_mae,
        "total_error_ms": total_error_ms,
        "error_count": error_count,

        
      
        "matched_count": len(covered_gt_errors_ms),
        "missed_gt_count": len(missed_gt_errors_ms),
        "false_pred_count": len(false_pred_errors_ms),

        "total_gt_segments": len(gt_segments),
        "total_pred_segments": len(pred_segments),

        
        "matched_mae": float(np.mean(covered_gt_errors_ms)) if covered_gt_errors_ms else 0.0,
        "missed_gt_mae": float(np.mean(missed_gt_errors_ms)) if missed_gt_errors_ms else 0.0,
        "false_pred_mae": float(np.mean(false_pred_errors_ms)) if false_pred_errors_ms else 0.0,
    }



def evaluate_temporal_performance(gt_probs, pred_probs, fps=50, threshold=0.5, min_iou=0.0):
    gt_probs = np.asarray(gt_probs)
    pred_probs = np.asarray(pred_probs)

    if gt_probs.ndim == 1:
        return evaluate_one_contact_channel(gt_probs, pred_probs, fps, threshold, min_iou)

    if gt_probs.ndim == 2:
        channel_results = []
        for ch in range(gt_probs.shape[1]):
            channel_results.append(
                evaluate_one_contact_channel(
                    gt_probs[:, ch],
                    pred_probs[:, ch],
                    fps,
                    threshold,
                    min_iou,
                )
            )

        total_error_ms = sum(r["total_error_ms"] for r in channel_results)
        error_count = sum(r["error_count"] for r in channel_results)
        matched_count = sum(r["matched_count"] for r in channel_results)
        missed_gt_count = sum(r["missed_gt_count"] for r in channel_results)
        false_pred_count = sum(r["false_pred_count"] for r in channel_results)
        return {
            "ct_mae": total_error_ms / error_count if error_count else 0.0,
            "total_error_ms": total_error_ms,
            "error_count": error_count,
            "matched_count": matched_count,
            "missed_gt_count": missed_gt_count,
            "false_pred_count": false_pred_count,
            "total_gt_segments": sum(r["total_gt_segments"] for r in channel_results),
            "total_pred_segments": sum(r["total_pred_segments"] for r in channel_results),
            "matched_mae": (
                sum(r["matched_mae"] * r["matched_count"] for r in channel_results)
                / matched_count
                if matched_count else 0.0
            ),
            "missed_gt_mae": (
                sum(r["missed_gt_mae"] * r["missed_gt_count"] for r in channel_results)
                / missed_gt_count
                if missed_gt_count else 0.0
            ),
            "false_pred_mae": (
                sum(r["false_pred_mae"] * r["false_pred_count"] for r in channel_results)
                / false_pred_count
                if false_pred_count else 0.0
            ),
        }

    raise ValueError(f"Unsupported contact shape: {gt_probs.shape}")

def save_summary_plots(subject_names, metrics_list, avg_metrics, fps):
    plot_names = subject_names + ["AVG"]

    ct_mae = [m["ct_mae"] for m in metrics_list] + [avg_metrics["ct_mae"]]
    matched_mae = [m["matched_mae"] for m in metrics_list] + [avg_metrics["matched_mae"]]
    missed_gt_mae = [m["missed_gt_mae"] for m in metrics_list] + [avg_metrics["missed_gt_mae"]]
    false_pred_mae = [m["false_pred_mae"] for m in metrics_list] + [avg_metrics["false_pred_mae"]]

    x = np.arange(len(plot_names))
    width = 0.2

    plt.figure(figsize=(14, 6))
    plt.bar(x - 1.5 * width, ct_mae, width, label="Contact Time MAE")
    plt.bar(x - 0.5 * width, matched_mae, width, label="Matched Duration MAE")
    plt.bar(x + 0.5 * width, missed_gt_mae, width, label="Missed GT Avg Length")
    plt.bar(x + 1.5 * width, false_pred_mae, width, label="False Pred Avg Length")

    plt.xticks(x, plot_names, rotation=45, ha="right")
    plt.ylabel("Milliseconds")
    plt.title(f"Contact Time MAE Breakdown @ {fps} FPS")
    plt.legend()
    plt.tight_layout()

    mae_plot = f"contact_time_mae_breakdown_{fps}fps.png"
    plt.savefig(mae_plot, dpi=300)
    plt.close()

    matched_count = [m["matched_count"] for m in metrics_list] + [avg_metrics["matched_count"]]
    missed_gt_count = [m["missed_gt_count"] for m in metrics_list] + [avg_metrics["missed_gt_count"]]
    false_pred_count = [m["false_pred_count"] for m in metrics_list] + [avg_metrics["false_pred_count"]]

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, matched_count, width, label="Matched Segments")
    plt.bar(x, missed_gt_count, width, label="Missed GT")
    plt.bar(x + width, false_pred_count, width, label="False Pred")

    plt.xticks(x, plot_names, rotation=45, ha="right")
    plt.ylabel("Segment Count")
    plt.title(f"Contact Segment Matching Counts @ {fps} FPS")
    plt.legend()
    plt.tight_layout()

    count_plot = f"contact_segment_counts_{fps}fps.png"
    plt.savefig(count_plot, dpi=300)
    plt.close()



def main():
    parser = argparse.ArgumentParser(
        description="Calculate coverage-based contact-time MAE"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="FPS of the video/model output, e.g. 10 or 50",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="output",
        help="Directory containing subject*_output.pkl files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize predicted contact probabilities",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.0,
        help="Minimum IoU required for a GT/predicted contact match",
    )
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.dir, "subject*_output.pkl")))

    if not pkl_files:
        print(f"No pkl files found in {args.dir}!")
        return

    metrics_list = []
    subject_names = []
    csv_lines = [
    "Subject,Contact_Time_MAE_ms,Total_Error_ms,Error_Count,"
    "Covered_GT_Segments,Missed_GT_FN,False_Pred_Outside_Pieces,Total_GT,Total_Pred,"
    "Matched_Duration_MAE_ms,Missed_GT_Avg_Length_ms,False_Pred_Avg_Length_ms\n"
    ]


    print("\n" + "=" * 72)
    print(
        f"Evaluating {len(pkl_files)} subjects at {args.fps} FPS "
        f"with segment IoU matching"
    )
    print("=" * 72)

    for pkl_path in pkl_files:
        subj_name = os.path.basename(pkl_path).split("_")[0]

        print(f"[{subj_name:>10}] 1/3 reading pkl...", end="", flush=True)
        gt, pred = extract_signals(pkl_path)
        if gt is None:
            print(" failed")
            continue

        print(" done | 2/3 evaluating segments...", end="", flush=True)
        res = evaluate_temporal_performance(
            gt,
            pred,
            fps=args.fps,
            threshold=args.threshold,
            min_iou=args.min_iou,
        )

        metrics_list.append(res)
        subject_names.append(subj_name)
        csv_lines.append(
            f"{subj_name},{res['ct_mae']:.2f},{res['total_error_ms']:.2f},"
            f"{res['error_count']},{res['matched_count']},"
            f"{res['missed_gt_count']},{res['false_pred_count']},"
            f"{res['total_gt_segments']},{res['total_pred_segments']},"
            f"{res['matched_mae']:.2f},{res['missed_gt_mae']:.2f},{res['false_pred_mae']:.2f}\n"
        )

        print(f" done | 3/3 Contact Time MAE: {res['ct_mae']:.2f} ms")

        del gt, pred, res
        gc.collect()

    if not metrics_list:
        print("No valid data processed.")
        return

    avg_metrics = {
        key: np.mean([metrics[key] for metrics in metrics_list])
        for key in metrics_list[0].keys()
    }
    csv_lines.append(
        f"AVG,{avg_metrics['ct_mae']:.2f},{avg_metrics['total_error_ms']:.2f},"
        f"{avg_metrics['error_count']:.1f},{avg_metrics['matched_count']:.1f},"
        f"{avg_metrics['missed_gt_count']:.1f},{avg_metrics['false_pred_count']:.1f},"
        f"{avg_metrics['total_gt_segments']:.1f},{avg_metrics['total_pred_segments']:.1f},"
        f"{avg_metrics['matched_mae']:.2f},{avg_metrics['missed_gt_mae']:.2f},"
        f"{avg_metrics['false_pred_mae']:.2f}\n"
    )

    print("\n" + "=" * 60)
    print(f"OVERALL CONTACT TIME REPORT @ {args.fps} FPS")
    print("-" * 60)
    print(f"{'Metric':<32} | {'Result':<12}")
    print("-" * 60)
    print(f"{'Contact Time MAE (ms)':<32} | {avg_metrics['ct_mae']:<12.2f}")
    print(f"{'Total Error (ms)':<32} | {avg_metrics['total_error_ms']:<12.2f}")
    print(f"{'Error Contributions':<32} | {avg_metrics['error_count']:<12.1f}")
    print("-" * 60)
    print(f"{'Avg Covered GT Segments':<32} | {avg_metrics['matched_count']:<12.1f}")
    print(f"{'Avg Missed GT Segments':<32} | {avg_metrics['missed_gt_count']:<12.1f}")
    print(f"{'Avg False Pred Outside Pieces':<32} | {avg_metrics['false_pred_count']:<12.1f}")
    print(f"{'Avg Total GT Segments':<32} | {avg_metrics['total_gt_segments']:<12.1f}")
    print(f"{'Avg Total Pred Segments':<32} | {avg_metrics['total_pred_segments']:<12.1f}")
    print("=" * 60)
    print(f"{'Matched Duration MAE (ms)':<32} | {avg_metrics['matched_mae']:<12.2f}")
    print(f"{'Missed GT Avg Length (ms)':<32} | {avg_metrics['missed_gt_mae']:<12.2f}")
    print(f"{'False Pred Avg Length (ms)':<32} | {avg_metrics['false_pred_mae']:<12.2f}")
    print("=" * 60)

    save_summary_plots(subject_names, metrics_list, avg_metrics, args.fps)
    csv_filename = f"contact_time_mae_results_{args.fps}fps.csv"
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.writelines(csv_lines)
    print(f"\nDetailed results saved to {csv_filename}")


if __name__ == "__main__":
    main()
