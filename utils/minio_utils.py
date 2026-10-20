import json
import logging

from minio import Minio

from config.minio_config import minio_config

logger = logging.getLogger(__name__)

minio_client = Minio(
    endpoint=minio_config.endpoint,
    access_key=minio_config.access_key,
    secret_key=minio_config.secret_key,
    secure=False,
)


def _ensure_bucket_and_policy() -> None:
    """确保桶存在，并设置公共读策略（便于 md 中直接引用图片 URL）。"""
    bucket = minio_config.bucket_name
    if not bucket:
        raise ValueError("MINIO_BUCKET_NAME 未配置")

    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)
        logger.info("已创建 MinIO 桶: %s", bucket)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/*"],
            }
        ],
    }
    minio_client.set_bucket_policy(bucket, json.dumps(policy))


_ensure_bucket_and_policy()


def get_minio_client() -> Minio:
    return minio_client


if __name__ == "__main__":
    client = get_minio_client()
    print("endpoint:", minio_config.endpoint)
    print("bucket:", minio_config.bucket_name, "exists=", client.bucket_exists(minio_config.bucket_name))
