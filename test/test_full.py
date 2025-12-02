import os
import boto3
import pytest
import pickle
import numpy as np

from pathlib import Path
from main import preprocess_bib_file, preprocess_txt_file
from botocore.exceptions import ClientError

MINIO_USER = "minioadmin"
MINIO_PWD = "minioadmin"
BUCKET_NAME = "testbucket"


def ensure_bucket(s3, bucket):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket"):
            s3.create_bucket(Bucket=bucket)
        else:
            raise


def download_to_tmp(s3, bucket, key):
    tmp_path = Path("/tmp") / key.replace("/", "_")
    s3.download_file(bucket, key, str(tmp_path))
    return tmp_path


@pytest.fixture
def s3_minio():
    client = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PWD
    )
    ensure_bucket(client, BUCKET_NAME)
    return client


def test_full_bib(s3_minio):
    input_file_name = "input"
    dtm_output_file_name = "dtm_file"
    vocab_output_file_name = "vocab_file"

    bib_path = Path(__file__).parent / "files" / f"{input_file_name}.bib"
    bib_bytes = bib_path.read_bytes()

    s3_minio.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{input_file_name}.bib",
        Body=bib_bytes
    )

    env = {
        "bib_file_S3_HOST": "http://127.0.0.1",
        "bib_file_S3_PORT": "9000",
        "bib_file_S3_ACCESS_KEY": MINIO_USER,
        "bib_file_S3_SECRET_KEY": MINIO_PWD,
        "bib_file_BUCKET_NAME": BUCKET_NAME,
        "bib_file_FILE_PATH": "",
        "bib_file_FILE_NAME": input_file_name,
        "bib_file_SELECTED_ATTRIBUTE": "abstract",

        "dtm_output_S3_HOST": "http://127.0.0.1",
        "dtm_output_S3_PORT": "9000",
        "dtm_output_S3_ACCESS_KEY": MINIO_USER,
        "dtm_output_S3_SECRET_KEY": MINIO_PWD,
        "dtm_output_BUCKET_NAME":  BUCKET_NAME,
        "dtm_output_FILE_PATH": "",
        "dtm_output_FILE_NAME": dtm_output_file_name,

        "vocab_output_S3_HOST": "http://127.0.0.1",
        "vocab_output_S3_PORT": "9000",
        "vocab_output_S3_ACCESS_KEY": MINIO_USER,
        "vocab_output_S3_SECRET_KEY": MINIO_PWD,
        "vocab_output_BUCKET_NAME": BUCKET_NAME,
        "vocab_output_FILE_PATH": "",
        "vocab_output_FILE_NAME": vocab_output_file_name,
    }

    for k, v in env.items():
        os.environ[k] = v

    preprocess_bib_file()

    keys = [
        o["Key"]
        for o in s3_minio.list_objects_v2(
            Bucket="testbucket").get("Contents", [])
    ]

    assert f"{dtm_output_file_name}.pkl" in keys
    assert f"{vocab_output_file_name}.pkl" in keys

    dtm_path = download_to_tmp(s3_minio, BUCKET_NAME, f"{
        dtm_output_file_name}.pkl")
    vocab_path = download_to_tmp(s3_minio, BUCKET_NAME, f"{
        vocab_output_file_name}.pkl")

    # Load produced results
    with open(dtm_path, "rb") as f:
        dtm = pickle.load(f)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load expected snapshot files
    expected_vocab_path = Path(__file__).parent / \
        "files" / "expected_vocab_from_bib.pkl"
    expected_dtm_path = Path(__file__).parent / "files" / \
        "expected_dtm_from_bib.pkl"

    with open(expected_vocab_path, "rb") as f:
        expected_vocab = pickle.load(f)

    with open(expected_dtm_path, "rb") as f:
        expected_dtm = pickle.load(f)

    assert vocab == expected_vocab
    np.testing.assert_array_equal(dtm, expected_dtm)


def test_full_txt(s3_minio):
    input_file_name = "input"
    dtm_output_file_name = "dtm_txt_file"
    vocab_output_file_name = "vocab_txt_file"

    txt_path = Path(__file__).parent / "files" / f"{input_file_name}.txt"
    txt_bytes = txt_path.read_bytes()

    s3_minio.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{input_file_name}.txt",
        Body=txt_bytes
    )

    env = {
        "txt_file_S3_HOST": "http://127.0.0.1",
        "txt_file_S3_PORT": "9000",
        "txt_file_S3_ACCESS_KEY": MINIO_USER,
        "txt_file_S3_SECRET_KEY": MINIO_PWD,
        "txt_file_BUCKET_NAME": BUCKET_NAME,
        "txt_file_FILE_PATH": "",
        "txt_file_FILE_NAME": input_file_name,

        "dtm_output_S3_HOST": "http://127.0.0.1",
        "dtm_output_S3_PORT": "9000",
        "dtm_output_S3_ACCESS_KEY": MINIO_USER,
        "dtm_output_S3_SECRET_KEY": MINIO_PWD,
        "dtm_output_BUCKET_NAME": BUCKET_NAME,
        "dtm_output_FILE_PATH": "",
        "dtm_output_FILE_NAME": dtm_output_file_name,

        "vocab_output_S3_HOST": "http://127.0.0.1",
        "vocab_output_S3_PORT": "9000",
        "vocab_output_S3_ACCESS_KEY": MINIO_USER,
        "vocab_output_S3_SECRET_KEY": MINIO_PWD,
        "vocab_output_BUCKET_NAME": BUCKET_NAME,
        "vocab_output_FILE_PATH": "",
        "vocab_output_FILE_NAME": vocab_output_file_name,
    }

    for k, v in env.items():
        os.environ[k] = v

    preprocess_txt_file()

    keys = [
        o["Key"]
        for o in s3_minio.list_objects_v2(
            Bucket=BUCKET_NAME).get("Contents", [])
    ]

    assert f"{dtm_output_file_name}.pkl" in keys
    assert f"{vocab_output_file_name}.pkl" in keys

    # Download produced files
    dtm_path = download_to_tmp(s3_minio, BUCKET_NAME, f"{
                               dtm_output_file_name}.pkl")
    vocab_path = download_to_tmp(s3_minio, BUCKET_NAME, f"{
                                 vocab_output_file_name}.pkl")

    # Load produced results
    with open(dtm_path, "rb") as f:
        dtm = pickle.load(f)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load expected snapshot files
    expected_vocab_path = Path(__file__).parent / \
        "files" / "expected_vocab_from_txt.pkl"
    expected_dtm_path = Path(__file__).parent / \
        "files" / "expected_dtm_from_txt.pkl"

    with open(expected_vocab_path, "rb") as f:
        expected_vocab = pickle.load(f)

    with open(expected_dtm_path, "rb") as f:
        expected_dtm = pickle.load(f)

    # Assertions
    assert vocab == expected_vocab

    if hasattr(dtm, "toarray"):
        np.testing.assert_array_equal(dtm.toarray(), expected_dtm.toarray())
    else:
        np.testing.assert_array_equal(dtm, expected_dtm)
