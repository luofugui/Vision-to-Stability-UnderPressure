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
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    try:
        if "targets" in data and "predictions" in data:
            gt_raw = np.asarray(data["targets"]["contact"])
            pred_raw = np.asarray(data["predictions"]["contact"])

            print(f"\n{os.path.basename(pkl_path)}")
            print(f"  GT contact shape:   {gt_raw.shape}")
            print(f"  Pred contact shape: {pred_raw.shape}")

            return gt_raw, pred_raw

        print(f"Available keys in pkl: {data.keys()}")
        raise KeyError("Cannot find contact keys in pkl.")
    except Exception as e:
        print(f"Error extracting signals: {e}")
        return None, None


def get_segments(binary_signal):
    binary_signal = np.asarray(binary_signal).astype(int)
    padded = np.concatenate(([0], binary_signal, [0]))
    diff = np.diff(padded)

    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    return list(zip(onsets, offsets))


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


def evaluate_one_contact_channel(gt_probs, pred_probs, fps=50, threshold=0.5):
    """
    Segment-level contact-time MAE.

    Matched error includes:
      1. GT duration not covered by any overlapping prediction
      2. prediction duration outside the matched GT segment boundary

    False prediction error only includes predicted segments that do not overlap
    any GT segment at all.
    """
    ms_per_frame = 1000.0 / fps

    gt_binary = (np.asarray(gt_probs) >= 0.5).astype(int)
    pred_binary = (np.asarray(pred_probs) > threshold).astype(int)

    gt_segments = get_segments(gt_binary)
    pred_segments = get_segments(pred_binary)

    matched_errors_ms = []
    missed_gt_errors_ms = []
    false_pred_errors_ms = []

    matched_pred_indices = set()

    for gt_seg in gt_segments:
        g_s, g_e = gt_seg
        gt_duration = g_e - g_s

        overlapping = []
        for pred_idx, pred_seg in enumerate(pred_segments):
            overlap = overlap_length(gt_seg, pred_seg)
            if overlap > 0:
                overlapping.append((pred_idx, pred_seg, overlap))
                matched_pred_indices.add(pred_idx)

        if not overlapping:
            missed_gt_errors_ms.append(gt_duration * ms_per_frame)
            continue

        pred_overlap_duration = sum(item[2] for item in overlapping)
        missing_inside_gt = max(0, gt_duration - pred_overlap_duration)

        outside_pred_duration = 0
        for _, pred_seg, _ in overlapping:
            outside_pieces = subtract_intervals(pred_seg, [gt_seg])
            outside_pred_duration += sum(
                piece_e - piece_s for piece_s, piece_e in outside_pieces
            )

        error_ms = (missing_inside_gt + outside_pred_duration) * ms_per_frame
        matched_errors_ms.append(error_ms)

    for pred_idx, pred_seg in enumerate(pred_segments):
        if pred_idx not in matched_pred_indices:
            p_s, p_e = pred_seg
            false_pred_errors_ms.append((p_e - p_s) * ms_per_frame)

    errors_ms = matched_errors_ms + missed_gt_errors_ms + false_pred_errors_ms

    error_count = len(errors_ms)
    total_error_ms = float(np.sum(errors_ms)) if error_count else 0.0
    ct_mae = total_error_ms / error_count if error_count else 0.0

    return {
        "ct_mae": ct_mae,
        "total_error_ms": total_error_ms,
        "error_count": error_count,

        "matched_count": len(matched_errors_ms),
        "missed_gt_count": len(missed_gt_errors_ms),
        "false_pred_count": len(false_pred_errors_ms),

        "total_gt_segments": len(gt_segments),
        "total_pred_segments": len(pred_segments),

        "matched_mae": float(np.mean(matched_errors_ms)) if matched_errors_ms else 0.0,
        "missed_gt_mae": float(np.mean(missed_gt_errors_ms)) if missed_gt_errors_ms else 0.0,
        "false_pred_mae": float(np.mean(false_pred_errors_ms)) if false_pred_errors_ms else 0.0,
    }


def evaluate_temporal_performance(gt_probs, pred_probs, fps=50, threshold=0.5):
    gt_probs = np.asarray(gt_probs)
    pred_probs = np.asarray(pred_probs)

    if gt_probs.ndim == 1:
        return evaluate_one_contact_channel(gt_probs, pred_probs, fps, threshold)

    if gt_probs.ndim == 2:
        channel_results = []
        for ch in range(gt_probs.shape[1]):
            channel_results.append(
                evaluate_one_contact_channel(
                    gt_probs[:, ch],
                    pred_probs[:, ch],
                    fps=fps,
                    threshold=threshold,
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
    plt.bar(x - 0.5 * width, matched_mae, width, label="Matched Boundary MAE")
    plt.bar(x + 0.5 * width, missed_gt_mae, width, label="Missed GT Avg Length")
    plt.bar(x + 1.5 * width, false_pred_mae, width, label="Unmatched False Pred Avg Length")

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
    plt.bar(x - width, matched_count, width, label="Matched GT Segments")
    plt.bar(x, missed_gt_count, width, label="Missed GT")
    plt.bar(x + width, false_pred_count, width, label="Unmatched False Pred")

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
        description="Calculate segment-level contact-time MAE"
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--dir", type=str, default="output")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.dir, "subject*_output.pkl")))

    if not pkl_files:
        print(f"No pkl files found in {args.dir}!")
        return

    metrics_list = []
    subject_names = []
    csv_lines = [
        "Subject,Contact_Time_MAE_ms,Total_Error_ms,Error_Count,"
        "Matched_GT_Segments,Missed_GT_FN,Unmatched_False_Pred,Total_GT,Total_Pred,"
        "Matched_Boundary_MAE_ms,Missed_GT_Avg_Length_ms,Unmatched_False_Pred_Avg_Length_ms\n"
    ]

    print("\n" + "=" * 72)
    print(f"Evaluating {len(pkl_files)} subjects at {args.fps} FPS")
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
    print(f"{'Metric':<40} | {'Result':<12}")
    print("-" * 60)
    print(f"{'Contact Time MAE (ms)':<40} | {avg_metrics['ct_mae']:<12.2f}")
    print(f"{'Total Error (ms)':<40} | {avg_metrics['total_error_ms']:<12.2f}")
    print(f"{'Error Contributions':<40} | {avg_metrics['error_count']:<12.1f}")
    print("-" * 60)
    print(f"{'Avg Matched GT Segments':<40} | {avg_metrics['matched_count']:<12.1f}")
    print(f"{'Avg Missed GT Segments':<40} | {avg_metrics['missed_gt_count']:<12.1f}")
    print(f"{'Avg Unmatched False Pred Segments':<40} | {avg_metrics['false_pred_count']:<12.1f}")
    print(f"{'Avg Total GT Segments':<40} | {avg_metrics['total_gt_segments']:<12.1f}")
    print(f"{'Avg Total Pred Segments':<40} | {avg_metrics['total_pred_segments']:<12.1f}")
    print("=" * 60)
    print(f"{'Matched Boundary MAE (ms)':<40} | {avg_metrics['matched_mae']:<12.2f}")
    print(f"{'Missed GT Avg Length (ms)':<40} | {avg_metrics['missed_gt_mae']:<12.2f}")
    print(f"{'Unmatched False Pred Avg Length (ms)':<40} | {avg_metrics['false_pred_mae']:<12.2f}")
    print("=" * 60)

    save_summary_plots(subject_names, metrics_list, avg_metrics, args.fps)

    csv_filename = f"contact_time_mae_results_{args.fps}fps.csv"
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.writelines(csv_lines)

    print(f"\nDetailed results saved to {csv_filename}")


if __name__ == "__main__":
    main()
