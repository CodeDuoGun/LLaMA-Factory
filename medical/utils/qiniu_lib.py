#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
七牛云文件管理类：支持本地文件上传、云端文件下载。
依赖：qiniu (pip install qiniu)、requests
"""

import os
import requests
from pathlib import Path
from qiniu import Auth, put_file_v2, BucketManager


class QiniuManager:
    """
    七牛云文件管理。

    Args:
        access_key: 七牛 Access Key
        secret_key: 七牛 Secret Key
        bucket_name: 存储空间名称
        bucket_domain: 存储空间绑定的域名，例如 'https://cdn.example.com'
    """

    def __init__(self, access_key: str, secret_key: str, bucket_name: str, bucket_domain: str):
        self.bucket_name = bucket_name
        self.bucket_domain = bucket_domain.rstrip("/")
        self._auth = Auth(access_key, secret_key)
        self._bucket = BucketManager(self._auth)

    # ------------------------------------------------------------------
    # 上传
    # ------------------------------------------------------------------

    def upload(self, local_path: str, key: str = None, overwrite: bool = True,
               token_expires: int = 3600) -> dict:
        """
        上传本地文件到七牛云。

        Args:
            local_path:     本地文件路径
            key:            云端保存的文件名；为 None 时使用本地文件名
            overwrite:      是否覆盖同名文件，默认 True
            token_expires:  上传 token 有效期（秒），默认 3600

        Returns:
            {'key': ..., 'hash': ..., 'url': ...}
        """
        local_path = Path(local_path).resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"文件不存在: {local_path}")

        if key is None:
            key = local_path.name

        policy = {"insertOnly": 0} if overwrite else {}
        token = self._auth.upload_token(self.bucket_name, key, token_expires, policy)

        ret, info = put_file_v2(token, key, str(local_path), version="v2")
        if info.status_code != 200:
            raise RuntimeError(f"上传失败: {info}")

        url = f"{self.bucket_domain}/{key}"
        print(f"[上传成功] {local_path.name}  ->  {url}")
        return {"key": ret["key"], "hash": ret["hash"], "url": url}

    def batch_upload(self, local_paths: list, key_prefix: str = "",
                     overwrite: bool = True) -> list:
        """
        批量上传本地文件。

        Args:
            local_paths: 本地文件路径列表
            key_prefix:  云端文件名前缀，例如 'videos/2026/'
            overwrite:   是否覆盖同名文件

        Returns:
            成功上传的结果列表
        """
        results = []
        for path in local_paths:
            try:
                key = key_prefix + Path(path).name
                result = self.upload(path, key=key, overwrite=overwrite)
                results.append(result)
            except Exception as e:
                print(f"[上传失败] {path}: {e}")
        return results

    # ------------------------------------------------------------------
    # 下载
    # ------------------------------------------------------------------

    def download(self, key: str, save_dir: str = ".", filename: str = None,
                 private: bool = True, expires: int = 3600) -> str:
        """
        从七牛云下载文件到本地。

        Args:
            key:      云端文件名
            save_dir: 本地保存目录，默认当前目录
            filename: 本地保存文件名；为 None 时使用 key 的末尾部分
            private:  是否为私有空间，默认 True（生成带签名的下载链接）
            expires:  私有链接有效期（秒），默认 3600

        Returns:
            本地文件绝对路径
        """
        base_url = f"{self.bucket_domain}/{key}"
        if private:
            download_url = self._auth.private_download_url(base_url, expires=expires)
        else:
            download_url = base_url

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        local_name = filename or Path(key).name
        save_path = save_dir / local_name

        print(f"[下载中] {key}  ->  {save_path}")
        with requests.get(download_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"[下载成功] {save_path}")
        return str(save_path.resolve())

    def batch_download(self, keys: list, save_dir: str = ".",
                       private: bool = True) -> list:
        """
        批量下载云端文件。

        Args:
            keys:     云端文件名列表
            save_dir: 本地保存目录
            private:  是否为私有空间

        Returns:
            成功下载的本地文件路径列表
        """
        results = []
        for key in keys:
            try:
                path = self.download(key, save_dir=save_dir, private=private)
                results.append(path)
            except Exception as e:
                print(f"[下载失败] {key}: {e}")
        return results

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        """检查云端文件是否存在。"""
        _, info = self._bucket.stat(self.bucket_name, key)
        if info.status_code == 200:
            return True
        if info.status_code == 612:
            return False
        raise RuntimeError(f"检查文件是否存在失败: {key} -> {info}")

    def get_url(self, key: str, private: bool = True, expires: int = 3600) -> str:
        """获取文件访问 URL（私有空间返回带签名链接）。"""
        base_url = f"{self.bucket_domain}/{key}"
        if private:
            return self._auth.private_download_url(base_url, expires=expires)
        return base_url

    def delete(self, key: str) -> None:
        """删除云端文件。"""
        ret, info = self._bucket.delete(self.bucket_name, key)
        if info.status_code != 200:
            raise RuntimeError(f"删除失败: {info}")
        print(f"[删除成功] {key}")


if __name__ == "__main__":
    # 使用示例
    manager = QiniuManager(
        access_key="jBs-KkX6p28ANYt1B_Rmcbw5T_fV-97zayA0nVgZ",
        secret_key="DruqdkFMq0VbyCTgTJ7bbbURClTzlqGqllW4zAQb",
        bucket_name="nlp-audio",
        bucket_domain="https://nlp-audio.sihuiyiliao.com",
    )

    # 上传单个文件
    result = manager.upload("./f0.wav", key="tcm/f0.wav")

    # 批量上传
    # manager.batch_upload(["./f0.wav"], key_prefix="tcm/")

    # 下载单个文件
    # manager.download("tcm/f0.wav", save_dir="./downloads")

    # 批量下载
    # manager.batch_download(["tcm/f0.wav"], save_dir="./downloads")
