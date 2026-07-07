#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 DashScope ASR transcription_url 的 JSON 内容，
将 sentences 转换为对话格式并存储到指定 JSON 文件。

对话格式：[{"speaker": <speaker_id>, "content": "<text>"}, ...]
"""

import json
import requests
from pathlib import Path


def parse_asr_result(asr_data: dict) -> list:
    """
    解析 ASR JSON 数据，提取对话列表。

    ASR 数据结构：
      transcripts[]
        └── sentences[]
              ├── speaker_id  (int)
              └── text        (str)

    相邻同说话人的句子会合并，避免过碎。

    Args:
        asr_data: transcription_url 返回的 JSON dict

    Returns:
        [{"speaker": int, "content": str}, ...]
    """
    sentences = []
    for transcript in asr_data.get("transcripts", []):
        sentences.extend(transcript.get("sentences", []))

    if not sentences:
        return []

    dialog = []
    for sent in sentences:
        speaker = sent.get("speaker_id", 0)
        text = sent.get("text", "").strip()
        begin_time = sent.get("begin_time")  # 毫秒
        end_time = sent.get("end_time")      # 毫秒
        if not text:
            continue
        # 合并相邻同说话人句子，更新 end_time
        if dialog and dialog[-1]["speaker"] == speaker:
            dialog[-1]["content"] += text
            dialog[-1]["end_time"] = end_time
        else:
            dialog.append({
                "speaker": speaker,
                "content": text,
                "begin_time": begin_time,
                "end_time": end_time,
            })

    return dialog


def transcription_url_to_dialog(transcription_url: str) -> list:
    """
    从 transcription_url 拉取 JSON，解析为对话列表。

    Args:
        transcription_url: ASR 结果的公网 URL

    Returns:
        对话列表
    """
    resp = requests.get(transcription_url, timeout=30)
    resp.raise_for_status()
    asr_data = resp.json()
    return parse_asr_result(asr_data)


def asr_response_to_dialog(transcribe_response: dict) -> list:
    """
    从 DashScope Transcription.fetch() 返回的 response 对象中
    读取 transcription_url 并转换为对话列表。

    Args:
        transcribe_response: Transcription.fetch() 的返回值（dict 或对象，
                             支持 .output 属性或 ["output"] 取值）

    Returns:
        对话列表
    """
    # 兼容 dict 和 对象两种形式
    if hasattr(transcribe_response, "output"):
        output = transcribe_response.output
    else:
        output = transcribe_response.get("output", {})

    results = output.get("results", [])
    if not results:
        raise ValueError("ASR response 中没有 results 字段")

    transcription_url = results[0].get("transcription_url", "")
    if not transcription_url:
        raise ValueError("ASR response 中没有 transcription_url 字段")

    return transcription_url_to_dialog(transcription_url)


def save_dialog(dialog: list, output_path: str) -> None:
    """
    将对话列表写入 JSON 文件。

    Args:
        dialog:      对话列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dialog, f, ensure_ascii=False, indent=2)
    print(f"[保存] 共 {len(dialog)} 条对话 -> {output_path}")


# ---------------------------------------------------------------------------
# 示例：直接从 transcription_url 转换
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # 用法1：python asr_to_dialog.py <transcription_url> [output.json]
    # 用法2：从本地 ASR JSON 文件读取
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) >= 3 else "test.json"

        if arg.startswith("http"):
            print(f"[拉取] {arg}")
            dialog = transcription_url_to_dialog(arg)
        else:
            # 本地 ASR JSON 文件
            print(f"[读取] {arg}")
            with open(arg, "r", encoding="utf-8") as f:
                asr_data = json.load(f)
            dialog = parse_asr_result(asr_data)

        save_dialog(dialog, out)
    else:
        # 演示：直接解析已有的 a91dff27-fee2-4abd-96ba-7e68a93ad9c7-1.json
        src = Path(__file__).parent.parent / "a91dff27-fee2-4abd-96ba-7e68a93ad9c7-1.json"
        out = Path(__file__).parent.parent / "test.json"
        print(f"[演示] 读取 {src}")
        with open(src, "r", encoding="utf-8") as f:
            asr_data = json.load(f)
        dialog = parse_asr_result(asr_data)
        save_dialog(dialog, str(out))
