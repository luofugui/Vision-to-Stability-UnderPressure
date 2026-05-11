import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_contact(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    gt = np.asarray(data["targets"]["contact"])
    pred_raw = np.asarray(data["predictions"]["contact"])

    is_logits = pred_raw.min() < 0 or pred_raw.max() > 1
    pred_prob = sigmoid(pred_raw) if is_logits else pred_raw

    return gt, pred_raw, pred_prob, is_logits


def get_event_indices(binary_signal):
    """
    onset: 0 -> 1
    departure: 1 -> 0

    Returns frame indices where event occurs.
    """
    x = np.asarray(binary_signal).astype(int)
    padded = np.concatenate(([0], x, [0]))
    diff = np.diff(padded)

    onsets = np.where(diff == 1)[0]
    departures = np.where(diff == -1)[0]

    return onsets, departures


def match_events(gt_events, pred_events, fps, max_match_ms=500):
    """
    Greedy nearest-neighbor event matching.

    Each GT event can match at most one pred event.
    Each pred event can match at most one GT event.
    """
    max_match_frames = int(round(max_match_ms / 1000.0 * fps))

    gt_events = np.asarray(gt_events, dtype=int)
    pred_events = np.asarray(pred_events, dtype=int)

    if len(gt_events) == 0:
        return [], [], list(range(len(pred_events)))

    if len(pred_events) == 0:
        return [], list(range(len(gt_events))), []

    candidates = []
    for gi, g in enumerate(gt_events):
        distances = np.abs(pred_events - g)
        close = np.where(distances <= max_match_frames)[0]
        for pi in close:
            candidates.append((distances[pi], gi, pi))

    candidates.sort(key=lambda x: x[0])

    matched = []
    used_gt = set()
    used_pred = set()

    for dist, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        matched.append((gt_events[gi], pred_events[pi], dist))
        used_gt.add(gi)
        used_pred.add(pi)

    missed_gt = [i for i in range(len(gt_events)) if i not in used_gt]
    false_pred = [i for i in range(len(pred_events)) if i not in used_pred]

    return matched, missed_gt, false_pred


def collect_event_windows(gt_binary, pred_prob, event_indices, fps, half_window_sec=0.5):
    half = int(round(half_window_sec * fps))
    win_len = 2 * half + 1

    gt_windows = []
    pred_windows = []

    for idx in event_indices:
        start = idx - half
        end = idx + half + 1

        if start < 0 or end > len(gt_binary):
            continue

        gt_windows.append(gt_binary[start:end])
        pred_windows.append(pred_prob[start:end])

    if not gt_windows:
        return None, None

    return np.stack(gt_windows), np.stack(pred_windows)


def evaluate_channel(gt, pred_raw, pred_prob, fps, threshold, max_match_ms, half_window_sec):
    gt_binary = (gt >= 0.5).astype(int)
    pred_binary = (pred_raw > threshold).astype(int)

    gt_onsets, gt_departures = get_event_indices(gt_binary)
    pred_onsets, pred_departures = get_event_indices(pred_binary)

    results = {}

    for event_name, gt_events, pred_events in [
        ("onset", gt_onsets, pred_onsets),
        ("departure", gt_departures, pred_departures),
    ]:
        matched, missed_gt, false_pred = match_events(
            gt_events,
            pred_events,
            fps=fps,
            max_match_ms=max_match_ms,
        )

        errors_ms = [abs(p - g) * 1000.0 / fps for g, p, _ in matched]

        gt_windows, pred_windows = collect_event_windows(
            gt_binary,
            pred_prob,
            gt_events,
            fps=fps,
            half_window_sec=half_window_sec,
        )

        results[event_name] = {
            "mae_ms": float(np.mean(errors_ms)) if errors_ms else 0.0,
            "median_ms": float(np.median(errors_ms)) if errors_ms else 0.0,
            "matched": len(matched),
            "missed_gt": len(missed_gt),
            "false_pred": len(false_pred),
            "gt_total": len(gt_events),
            "pred_total": len(pred_events),
            "errors_ms": errors_ms,
            "gt_windows": gt_windows,
            "pred_windows": pred_windows,
        }

    return results


def aggregate_results(channel_results):
    summary = {}

    for event_name in ["onset", "departure"]:
        all_errors = []
        matched = missed = false = gt_total = pred_total = 0

        for res in channel_results:
            r = res[event_name]
            all_errors.extend(r["errors_ms"])
            matched += r["matched"]
            missed += r["missed_gt"]
            false += r["false_pred"]
            gt_total += r["gt_total"]
            pred_total += r["pred_total"]

        summary[event_name] = {
            "mae_ms": float(np.mean(all_errors)) if all_errors else 0.0,
            "median_ms": float(np.median(all_errors)) if all_errors else 0.0,
            "matched": matched,
            "missed_gt": missed,
            "false_pred": false,
            "gt_total": gt_total,
            "pred_total": pred_total,
        }

    return summary


def plot_event_windows(all_windows, fps, half_window_sec, out_prefix):
    half = int(round(half_window_sec * fps))
    t = (np.arange(2 * half + 1) - half) / fps

    for event_name in ["onset", "departure"]:
        gt_list = []
        pred_list = []

        for item in all_windows[event_name]:
            gt_w, pred_w = item
            if gt_w is not None:
                gt_list.append(gt_w)
                pred_list.append(pred_w)

        if not gt_list:
            continue

        gt_all = np.concatenate(gt_list, axis=0)
        pred_all = np.concatenate(pred_list, axis=0)

        gt_mean = gt_all.mean(axis=0)
        pred_mean = pred_all.mean(axis=0)
        pred_std = pred_all.std(axis=0)

        plt.figure(figsize=(8, 4))
        plt.plot(t, gt_mean, label="GT contact", linewidth=2)
        plt.plot(t, pred_mean, label="Pred contact probability", linewidth=2)
        plt.fill_between(
            t,
            pred_mean - pred_std,
            pred_mean + pred_std,
            alpha=0.2,
            label="Pred +/- std",
        )
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Time relative to GT event (s)")
        plt.ylabel("Contact / probability")
        plt.title(f"{event_name.capitalize()} event window +/- {half_window_sec}s")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{event_name}_event_window.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-match-ms", type=float, default=500.0)
    parser.add_argument("--half-window-sec", type=float, default=0.5)
    parser.add_argument("--out-prefix", type=str, default="event_timing")
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.dir, "subject*_output.pkl")))
    if not pkl_files:
        print(f"No subject*_output.pkl files found in {args.dir}")
        return

    csv_lines = [
        "Subject,Event,MAE_ms,Median_ms,Matched,Missed_GT,False_Pred,GT_Total,Pred_Total\n"
    ]

    all_subject_summaries = []
    all_windows = {"onset": [], "departure": []}

    for pkl_path in pkl_files:
        subject = os.path.basename(pkl_path).replace("_output.pkl", "")
        gt, pred_raw, pred_prob, is_logits = load_contact(pkl_path)

        print(f"\n{subject}")
        print(f"  GT shape: {gt.shape}")
        print(f"  Pred shape: {pred_raw.shape}")
        print(f"  Pred type: {'logits' if is_logits else 'probabilities'}")
        print(f"  Threshold: {args.threshold}")

        channel_results = []

        for ch in range(gt.shape[1]):
            res = evaluate_channel(
                gt[:, ch],
                pred_raw[:, ch],
                pred_prob[:, ch],
                fps=args.fps,
                threshold=args.threshold,
                max_match_ms=args.max_match_ms,
                half_window_sec=args.half_window_sec,
            )
            channel_results.append(res)

            for event_name in ["onset", "departure"]:
                all_windows[event_name].append(
                    (res[event_name]["gt_windows"], res[event_name]["pred_windows"])
                )

        summary = aggregate_results(channel_results)
        all_subject_summaries.append(summary)

        for event_name in ["onset", "departure"]:
            r = summary[event_name]
            print(
                f"  {event_name:<10} MAE={r['mae_ms']:.2f} ms | "
                f"median={r['median_ms']:.2f} ms | "
                f"matched={r['matched']} missed={r['missed_gt']} false={r['false_pred']}"
            )
            csv_lines.append(
                f"{subject},{event_name},{r['mae_ms']:.2f},{r['median_ms']:.2f},"
                f"{r['matched']},{r['missed_gt']},{r['false_pred']},"
                f"{r['gt_total']},{r['pred_total']}\n"
            )

    print("\n" + "=" * 70)
    print(f"OVERALL EVENT TIMING REPORT @ {args.fps} FPS")
    print("=" * 70)

    for event_name in ["onset", "departure"]:
        maes = [s[event_name]["mae_ms"] for s in all_subject_summaries]
        medians = [s[event_name]["median_ms"] for s in all_subject_summaries]
        print(
            f"{event_name:<10} mean MAE={np.mean(maes):.2f} ms | "
            f"std={np.std(maes):.2f} ms | "
            f"mean median={np.mean(medians):.2f} ms"
        )

    csv_path = f"{args.out_prefix}_event_timing_results_{args.fps}fps.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(csv_lines)

    plot_event_windows(
        all_windows,
        fps=args.fps,
        half_window_sec=args.half_window_sec,
        out_prefix=args.out_prefix,
    )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plots: {args.out_prefix}_onset_event_window.png")
    print(f"Saved plots: {args.out_prefix}_departure_event_window.png")


if __name__ == "__main__":
    main()
