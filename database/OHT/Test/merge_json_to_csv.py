
import os
import glob
import json
import argparse
import pandas as pd

SENSOR_KEYS = ["NTC", "PM10", "PM2.5", "PM1.0", "CT1", "CT2", "CT3", "CT4"]

OUT_COLS = [
    "NTC", "PM10", "PM2.5", "PM1.0", "CT1", "CT2", "CT3", "CT4",
    "value_TGmx", "X_Tmax", "Y_Tmax",
    "ex_temperature", "ex_humidity", "ex_illuminance",
    "collection_datetime", "tagging_state",
]


def _safe_get_first(d, key, value_field="value"):
    try:
        arr = d.get(key)
        if not arr:
            return None
        obj = arr[0] if isinstance(arr, list) and len(arr) > 0 else None
        if not isinstance(obj, dict):
            return None
        return obj.get(value_field)
    except Exception:
        return None


def parse_one_json(fp: str) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        j = json.load(f)

    meta = (j.get("meta_info") or [{}])[0]
    sensor_block = (j.get("sensor_data") or [{}])[0]
    ir_block = (j.get("ir_data") or [{}])[0]
    ann_block = (j.get("annotations") or [{}])[0]
    ext_block = (j.get("external_data") or [{}])[0]

    row = {c: None for c in OUT_COLS}

    # 8 sensors
    for k in SENSOR_KEYS:
        row[k] = _safe_get_first(sensor_block, k, "value")

    # IR temp_max
    try:
        temp_max = (ir_block.get("temp_max") or [{}])[0]
        if isinstance(temp_max, dict):
            row["value_TGmx"] = temp_max.get("value_TGmx")
            row["X_Tmax"] = temp_max.get("X_Tmax")
            row["Y_Tmax"] = temp_max.get("Y_Tmax")
    except Exception:
        pass

    # external
    row["ex_temperature"] = _safe_get_first(ext_block, "ex_temperature", "value")
    row["ex_humidity"] = _safe_get_first(ext_block, "ex_humidity", "value")
    row["ex_illuminance"] = _safe_get_first(ext_block, "ex_illuminance", "value")

    # datetime = collection_date + collection_time
    cdate = meta.get("collection_date")
    ctime = meta.get("collection_time")
    row["collection_datetime"] = f"{cdate} {ctime}" if (cdate and ctime) else None

    # tagging state
    try:
        tagging = (ann_block.get("tagging") or [{}])[0]
        if isinstance(tagging, dict):
            row["tagging_state"] = tagging.get("state")
    except Exception:
        pass

    return row


def merge_folder(folder_path: str, out_csv_path: str, pattern="*.json") -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not files:
        return (0, 0)

    rows = []
    for fp in files:
        try:
            rows.append(parse_one_json(fp))
        except Exception as e:
            print(f"[WARN] skip file: {fp} ({e})")

    df = pd.DataFrame(rows, columns=OUT_COLS)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return (len(df), df.shape[1])


def main():
    ap = argparse.ArgumentParser(description="Merge Label subfolders' json files into CSVs saved under Data/")
    ap.add_argument("--root", default=".", help="프로젝트 루트 (기본: 현재 경로)")
    ap.add_argument("--json_dir", default="json", help="Label 폴더명/경로")
    ap.add_argument("--data_dir", default="Data", help="Data 폴더명/경로 (출력)")
    ap.add_argument("--pattern", default="*.json", help="json 파일 패턴")
    ap.add_argument("--recursive", action="store_true",
                    help="Label 아래 1-depth가 아닌 더 깊은 폴더도 탐색")
    args = ap.parse_args()

    json_root = os.path.join(args.root, args.json_dir)
    data_root = os.path.join(args.root, args.data_dir)

    if not os.path.isdir(json_root):
        raise SystemExit(f"Label dir not found: {json_root}")

    # 대상 폴더 리스트 (Label 바로 아래 폴더들)
    if args.recursive:
        # Label 아래 모든 하위폴더 중 json을 가진 폴더를 대상으로
        candidate_dirs = []
        for d, _, _ in os.walk(json_root):
            if glob.glob(os.path.join(d, args.pattern)):
                candidate_dirs.append(d)
        candidate_dirs = sorted(candidate_dirs)
    else:
        candidate_dirs = sorted(
            [os.path.join(json_root, name) for name in os.listdir(json_root)]
        )
        candidate_dirs = [d for d in candidate_dirs if os.path.isdir(d)]

    if not candidate_dirs:
        raise SystemExit(f"No subfolders found under: {json_root}")

    total = 0
    for folder in candidate_dirs:
        folder_name = os.path.basename(folder.rstrip("/"))
        out_csv = os.path.join(data_root, f"{folder_name}.csv")

        nrows, ncols = merge_folder(folder, out_csv, pattern=args.pattern)
        if nrows == 0:
            print(f"[SKIP] {folder_name}: no json files")
            continue

        total += 1
        print(f"[OK] {folder_name} -> {out_csv} (rows={nrows}, cols={ncols})")

    print(f"Done. created {total} csv files in {data_root}")


if __name__ == "__main__":
    main()