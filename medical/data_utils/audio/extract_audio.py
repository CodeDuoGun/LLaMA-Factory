#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从视频文件中提取音频，输出到与视频同级目录。
依赖：ffmpeg-python (pip install ffmpeg-python)，系统需安装 ffmpeg。
"""

import os
import sys
import traceback
import json
import tempfile
import subprocess
import argparse
import shutil
from http import HTTPStatus
from pathlib import Path
import requests

import ffmpeg
import dashscope
from dashscope.audio.asr import Transcription
from medical.data_utils.log import logger
from medical.data_utils.audio.asr_to_dialog import asr_response_to_dialog
from medical.utils.qiniu_lib import QiniuManager
from medical.data_utils.audio.db import check_audio_file_exists, insert_audio_label
from dotenv import load_dotenv

load_dotenv()

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

QINIU_ACCESS_KEY = os.getenv("QINIU_ACCESS_KEY")
QINIU_SECRET_KEY = os.getenv("QINIU_SECRET_KEY")
QINIU_BUCKET = "nlp-audio"
QINIU_DOMAIN = "https://nlp-audio.sihuiyiliao.com"
QINIU_KEY_PREFIX = "tcm/"
DEFAULT_DATA_DIR = Path("medical/data")
DEFAULT_JSON_FILE = DEFAULT_DATA_DIR / "wuweiping_record_20260525.json"


def normalize_doctorname(doctorname: str) -> str:
    doctorname = doctorname.strip().strip("/")
    if doctorname.startswith("online_"):
        doctorname = doctorname.removeprefix("online_")
    if not doctorname or "/" in doctorname:
        raise ValueError("doctorname 不能为空，且不能包含路径分隔符")
    return doctorname


def resolve_ffmpeg_binary(ffmpeg_bin: str | None = None) -> str:
    ffmpeg_bin = ffmpeg_bin or os.getenv("FFMPEG_BINARY") or "ffmpeg"
    resolved = shutil.which(ffmpeg_bin)
    if resolved:
        return resolved

    if Path(ffmpeg_bin).exists():
        return ffmpeg_bin

    raise FileNotFoundError(
        "未找到 ffmpeg 可执行文件。请先安装 ffmpeg，或通过 --ffmpeg-bin / FFMPEG_BINARY 指定路径，"
        "例如：--ffmpeg-bin /opt/homebrew/bin/ffmpeg"
    )


def download_video(url: str, save_path, max_retries: int = 3) -> str:
    """
    从 HTTP(S) URL 下载视频文件到本地路径，支持重试。

    Args:
        url:        视频的公网 URL
        save_path:  本地保存路径（str 或 Path）
        max_retries: 最大重试次数

    Returns:
        保存文件的绝对路径字符串
    """
    import time
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[下载视频] {url}  ->  {save_path.name}  (attempt {attempt})")
            with requests.get(url, stream=True, timeout=(30, 600)) as resp:
                resp.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
            size_mb = save_path.stat().st_size / 1024 / 1024
            logger.debug(f"[下载完成] {save_path.name}  ({size_mb:.1f} MB)")
            return str(save_path.resolve())
        except Exception as e:
            last_err = e
            logger.warning(f"[下载失败] attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(5)
    raise RuntimeError(f"视频下载失败（已重试 {max_retries} 次）: {last_err}")


def extract_wav_16k(input_path: str, output_path: str = None, ffmpeg_bin: str | None = None) -> str:
    """
    从视频/音频中提取 16kHz 单声道 PCM WAV，并做响度归一化。

    Args:
        input_path:  输入文件路径
        output_path: 输出 WAV 路径；为 None 时与输入同名，仅扩展名改为 .wav

    Returns:
        输出文件的绝对路径字符串
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    cmd = [
        resolve_ffmpeg_binary(ffmpeg_bin), "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", "loudnorm",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    logger.debug(f"[音频提取] {input_path.name}  ->  {output_path.name}")
    return str(output_path.resolve())


def extract_audio(video_file: str, audio_ext: str = "wav", ffmpeg_bin: str | None = None) -> str:
    """
    从单个视频文件提取音频（输出到视频同级目录）。

    Returns:
        输出音频文件的绝对路径
    """
    video_path = Path(video_file).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    output_path = video_path.with_suffix(f".{audio_ext}")

    (
        ffmpeg
        .input(str(video_path))
        .output(str(output_path), vn=None,
                acodec="libmp3lame" if audio_ext == "mp3" else "copy",
                loglevel="error")
        .overwrite_output()
        .run(cmd=resolve_ffmpeg_binary(ffmpeg_bin))
    )

    logger.debug(f"[OK] {video_path.name}  ->  {output_path.name}")
    return str(output_path)

def parse_transcript(resp):

    data = []
    results = resp.output.get("results", [])

    for result in results:
        transcripts = result.get("transcripts", [])
        for t in transcripts:
            sentences = t.get("sentences", [])
            for s in sentences:
                speaker = s.get("speaker_id", None)
                text = s.get("text", "").strip()

                if text:
                    data.append({
                        "speaker": speaker,
                        "content": text
                    })

    return data

def call_asr(file_url: str) -> str:
    """
    调用阿里云 DashScope ASR 对音频 URL 进行识别。

    Args:
        file_url: 音频文件的公网 URL（七牛云上传后的地址）

    Returns:
        识别出的纯文本字符串
    """
    try:
        resp = Transcription.async_call(
            model="fun-asr-2025-11-07",
            file_urls=[file_url],
            diarization_enabled=True,
            language_hints=["zh"],
            # speaker_count=2,
        )
        print(resp)
        # TODO 这里会阻塞主进程
        while resp.output.task_status not in ("SUCCEEDED", "FAILED"):
            resp = Transcription.fetch(task=resp.output.task_id)
        print(resp)
        if resp.status_code != HTTPStatus.OK or resp.output.task_status == "FAILED":
            raise RuntimeError(f"ASR 失败: {resp}")
        data = asr_response_to_dialog(resp)
        return data
    except Exception as e:
        logger.error(traceback.format_exc())
        return ""


def save_json(data, path="output.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_badcase(record: dict, path="badcase.jsonl"):
    """追加写入一行 JSON Lines 格式的 badcase 记录"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_qiniu_key_prefix(doctorname: str | None = None) -> str:
    if doctorname is None:
        return QINIU_KEY_PREFIX

    doctorname = normalize_doctorname(doctorname)
    return f"{QINIU_KEY_PREFIX}{doctorname}/"


def load_qiniu_json(qiniu: QiniuManager, key: str, save_dir: Path):
    json_path = qiniu.download(key, save_dir=str(save_dir), private=True)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_process(video_files: dict, doctorname: str | None = None, ffmpeg_bin: str | None = None) -> list:
    """
    批量处理视频文件：
      1. 提取 16kHz WAV 到临时目录
      2. 上传 WAV 到七牛云，获取公网 URL
      3. 调用 ASR 识别，结果写入同名 TXT 文件（临时目录）
      4. 上传 TXT 到七牛云
      5. 返回每个文件的处理结果

    Args:
        video_files: 视频文件路径列表

    Returns:
        处理结果列表，每项包含 audio_url / txt_url / text
    """
    qiniu_key_prefix = build_qiniu_key_prefix(doctorname)

    qiniu = QiniuManager(
        access_key=QINIU_ACCESS_KEY,
        secret_key=QINIU_SECRET_KEY,
        bucket_name=QINIU_BUCKET,
        bucket_domain=QINIU_DOMAIN,
    )

    results = []

    with tempfile.TemporaryDirectory(prefix="doctordigital_asr_") as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for idx, video_files in video_files.items():
            for video_file in video_files:
                stem = video_file.split("/")[-2]
                logger.debug(f"file stem: {stem}")

                audio_key = f"{qiniu_key_prefix}{idx}_{stem}.wav"
                json_key = f"{qiniu_key_prefix}{idx}_{stem}.json"
                audio_url = f"{QINIU_DOMAIN}/{audio_key}"
                json_url = f"{QINIU_DOMAIN}/{json_key}"
                audio_exists = qiniu.exists(audio_key)
                json_exists = qiniu.exists(json_key)
                db_exists = check_audio_file_exists(audio_url)

                if audio_exists and json_exists:
                    if db_exists:
                        logger.debug(f"[SKIP] 七牛云和 audio_lable 中已存在 {audio_key}，跳过")
                        results.append({
                            "video": str(video_file),
                            "audio_key": audio_key,
                            "audio_url": audio_url,
                            "txt_key": json_key,
                            "txt_url": json_url,
                            "skipped": True,
                        })
                        continue

                    json_data = load_qiniu_json(qiniu, json_key, tmp_dir)
                    insert_audio_label(audio_url, json_data)
                    logger.debug(f"[DB] 已从七牛云 JSON 回填 audio_lable: {audio_key}")
                    results.append({
                        "video": str(video_file),
                        "audio_key": audio_key,
                        "audio_url": audio_url,
                        "txt_key": json_key,
                        "txt_url": json_url,
                        "json_data": json_data,
                        "restored_db": True,
                    })
                    continue

                wav_path = tmp_dir / f"{idx}_{stem}.wav"
                json_path = tmp_dir / f"{idx}_{stem}.json"
                result_video = str(video_file)

                try:
                    if audio_exists:
                        logger.debug(f"[SKIP] 七牛云已存在音频 {audio_key}，复用")
                    else:
                        # 先落盘 video 到临时文件，再提取 wav 上传。
                        video_path = tmp_dir / f"{idx}_{stem}.mp4"
                        result_video = str(video_path)
                        if str(video_file).startswith("http"):
                            download_video(video_file, video_path)
                        else:
                            video_path = Path(video_file)
                            result_video = str(video_path)
                        resolved_ffmpeg_bin = resolve_ffmpeg_binary(ffmpeg_bin)
                        extract_wav_16k(str(video_path), str(wav_path), ffmpeg_bin=resolved_ffmpeg_bin)
                        audio_ret = qiniu.upload(str(wav_path), key=audio_key, overwrite=True)
                        audio_url = audio_ret["url"]

                    logger.debug(f"[ASR] 开始识别: {audio_url}")
                    json_data = call_asr(audio_url)
                    logger.debug(f"[ASR] 识别完成: {stem}，共 {len(json_data)} 字")

                    # save_json(json_data, "multi_person.json")
                    save_json(json_data, json_path)

                    json_ret = qiniu.upload(str(json_path), key=json_key, overwrite=True)
                    json_url = json_ret["url"]

                    if db_exists:
                        logger.debug(f"[DB] audio_lable 中已存在 {audio_key}，不重复插入")
                    else:
                        insert_audio_label(audio_url, json_data)
                        logger.debug(f"[DB] 已插入 audio_lable: {audio_key}")

                    results.append({
                        "video": result_video,
                        "audio_key": audio_key,
                        "audio_url": audio_url,
                        "txt_key": json_key,
                        "txt_url": json_url,
                        "json_data": json_data,
                    })

                except Exception as e:
                    logger.error(f"[FAIL] {video_file}: {traceback.format_exc()}")
                    save_badcase({
                        "idx": idx,
                        "stem": stem,
                        "video_file": video_file,
                        "audio_key": audio_key,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                    results.append({"video": result_video, "error": str(e)})

    return results


def get_immsg_videos(json_file: str):
    """"""
    with open(json_file, "r", encoding="utf-8") as fp:
        im_data = json.load(fp)
    videos = {}
    for idxdata in im_data:
        videos[idxdata["id"]] = []
        for msg in idxdata["im_msg"]:
            if msg["type"] == "video":
                videos[idxdata["id"]].append(msg["content"])
    return videos


def discover_online_json_files(doctorname: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[Path]:
    doctorname = normalize_doctorname(doctorname)

    online_dir = data_dir / f"online_{doctorname}"
    if online_dir.is_dir():
        files = sorted(path for path in online_dir.glob("*.json") if not path.name.startswith("._"))
        if files:
            return files

    files = sorted(path for path in data_dir.glob(f"*{doctorname}*.json") if not path.name.startswith("._"))
    if files:
        return files

    raise FileNotFoundError(f"未找到 online 医生数据: {online_dir}/*.json")


def get_immsg_videos_from_files(json_files: list[Path]) -> dict:
    video_dict = {}
    for json_file in json_files:
        for record_id, videos in get_immsg_videos(str(json_file)).items():
            if record_id in video_dict:
                video_dict[record_id].extend(videos)
            else:
                video_dict[record_id] = videos
    return video_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取问诊视频音频、ASR 后上传到七牛云。")
    parser.add_argument("--doctorname", help="医生数据标识，如 mashengmin；会读取 medical/data/online_<doctorname>/*.json。")
    parser.add_argument("--json-file", type=Path, default=None, help="显式指定问诊 JSON 文件；优先级高于 --doctorname。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="online 医生数据所在根目录。")
    parser.add_argument("--ffmpeg-bin", help="ffmpeg 可执行文件路径；也可用环境变量 FFMPEG_BINARY 指定。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.json_file:
        json_files = [args.json_file]
    elif args.doctorname:
        json_files = discover_online_json_files(args.doctorname, args.data_dir)
    else:
        json_files = [DEFAULT_JSON_FILE]

    logger.debug(f"读取问诊数据: {', '.join(str(path) for path in json_files)}")
    video_dict = get_immsg_videos_from_files(json_files)
    # video_dict = {"001": ["https://1305071164.vod2.myqcloud.com/0500fb7evodcq1305071164/c47598785145403701725920372/f0.mp4"]}

    logger.debug(f"共 {len(video_dict)} 个视频文件，开始处理...")

    all_results = batch_process(video_dict, doctorname=args.doctorname, ffmpeg_bin=args.ffmpeg_bin)
