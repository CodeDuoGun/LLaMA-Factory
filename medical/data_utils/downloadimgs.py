"""
批量下载 medical/data/wuweiping_record_20260525.json 中的图片。
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from tqdm import tqdm

DATA_FILE = Path(__file__).parent.parent / "data" / "wuweiping_record_20260525.json"
TONGUE_FACE_DIR = Path(__file__).parent.parent / "data" / "tongue_face"
REPORT_DIR = Path(__file__).parent.parent / "data" / "report"
MAX_WORKERS = 16
TIMEOUT = 30


def collect_tasks():
    tasks = {"tongue_face": [], "report": []}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        record_id = record.get("id", 0)
        order_sn = record.get("order_sn", "")

        for i, img_entry in enumerate(record.get("tongue_face_img", []), 1):
            if url := img_entry.get("img", "").strip():
                tasks["tongue_face"].append(
                    {"url": url, "index": len(tasks["tongue_face"]) + 1}
                )

        for i, img_entry in enumerate(record.get("admin_report_img", []), 1):
            if url := img_entry.get("img", "").strip():
                tasks["report"].append(
                    {"url": url, "index": len(tasks["report"]) + 1}
                )

    return tasks


def download_image(task, dest_dir, session):
    url = task["url"]
    index = task["index"]

    parsed = urlparse(url)
    original_name = unquote(parsed.path).split("/")[-1]
    name_part, ext = (original_name.split(".")[0], original_name.split(".")[1])
    filename = f"{name_part}_{index:05d}.{ext}"
    filepath = dest_dir / filename

    if filepath.exists():
        return True, filepath

    try:
        response = session.get(url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True, filepath
    except Exception:
        if filepath.exists():
            filepath.unlink()
        return False, url


def download_batch(tasks, dest_dir, desc):
    import concurrent.futures

    dest_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    success, failed = 0, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, t, dest_dir, session): t for t in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc=desc,
            unit="img",
        ):
            ok, result = future.result()
            if ok:
                success += 1
            else:
                failed.append(result)

    session.close()
    return success, failed


def main():
    if not DATA_FILE.exists():
        print(f"数据文件不存在: {DATA_FILE}")
        sys.exit(1)

    tasks = collect_tasks()
    print(f"tongue_face: {len(tasks['tongue_face'])} 张, report: {len(tasks['report'])} 张")

    tongue_ok, tongue_fail = download_batch(
        tasks["tongue_face"], TONGUE_FACE_DIR, "tongue_face"
    )
    print(f"tongue_face: {tongue_ok} 成功, {len(tongue_fail)} 失败")

    report_ok, report_fail = download_batch(tasks["report"], REPORT_DIR, "report")
    print(f"report: {report_ok} 成功, {len(report_fail)} 失败")

    total_ok = tongue_ok + report_ok
    total_fail = len(tongue_fail) + len(report_fail)
    print(f"总计: {total_ok} 成功, {total_fail} 失败")

    if total_fail:
        error_file = Path(__file__).parent / "download_errors.txt"
        with open(error_file, "w") as f:
            for url in tongue_fail + report_fail:
                f.write(url + "\n")
        print(f"失败列表: {error_file}")


if __name__ == "__main__":
    main()
