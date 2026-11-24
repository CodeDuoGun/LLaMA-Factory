import os
from pathlib import Path

import oss2


bucket = oss2.Bucket(
    oss2.Auth(
        os.environ["OSS_ACCESS_KEY_ID"],
        os.environ["OSS_ACCESS_KEY_SECRET"],
    ),
    os.environ["OSS_ENDPOINT"],
    os.environ["OSS_BUCKET"],
)


def sync_folder_to_oss(
    local_dir: str,
    oss_prefix: str,
    delete_removed: bool = False,
):
    """
    将本地目录同步至 OSS。

    1. 新增文件：直接上传
    2. 同名文件：覆盖 OSS 原文件
    3. 删除文件：按参数决定是否删除 OSS 对应文件
    """
    local_dir = Path(local_dir).resolve()
    oss_prefix = oss_prefix.strip("/") + "/"

    if not local_dir.is_dir():
        raise ValueError(f"本地目录不存在: {local_dir}")

    if oss_prefix == "/":
        raise ValueError("禁止同步至 Bucket 根目录")

    local_files = {
        oss_prefix + file.relative_to(local_dir).as_posix(): file
        for file in local_dir.rglob("*")
        if file.is_file() and not file.is_symlink()
    }

    # 防止本地目录异常为空，导致远端文件全部删除
    if not local_files:
        raise ValueError("本地目录为空，终止同步")

    # 1. 获取 OSS 当前目录下的全部文件
    remote_files = {
        obj.key
        for obj in oss2.ObjectIterator(
            bucket, prefix=oss_prefix
        )
        if not obj.key.endswith("/")
    }

    # 2. 上传全部本地文件，同名文件直接覆盖
    for oss_key, local_path in local_files.items():
        bucket.put_object_from_file(
            oss_key,
            str(local_path),
        )

        print(f"上传成功: {oss_key}")

    # 3. 删除 OSS 中多余的文件
    if delete_removed:
        removed_files = remote_files - local_files.keys()

        for oss_key in removed_files:
            bucket.delete_object(oss_key)
            print(f"删除成功: {oss_key}")

    print("OSS 目录同步完成")


if __name__ == "__main__":
    sync_folder_to_oss(
        local_dir="medical/data/二期南区/piaobingkui",
        oss_prefix="tcm-workbench/raw-data/",
        delete_removed=True,
    )