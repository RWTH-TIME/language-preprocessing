import os
import boto3
import pytest
import pandas as pd

from pathlib import Path
from main import preprocess_bib_file, preprocess_txt_file
from botocore.exceptions import ClientError
from sqlalchemy import create_engine

MINIO_USER = "minioadmin"
MINIO_PWD = "minioadmin"
BUCKET_NAME = "testbucket"

PG_USER = "postgres"
PG_PASS = "postgres"


def parse_pg_array(arr: str) -> list[str]:
    # Convert Postgres literal → Python list
    arr = arr.strip("{}")
    if not arr:
        return []
    # handle quoted items
    return [a.strip('"') for a in arr.split(",")]


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

    bib_path = Path(__file__).parent / "files" / f"{input_file_name}.bib"
    bib_bytes = bib_path.read_bytes()

    # Upload to MinIO
    s3_minio.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{input_file_name}.bib",
        Body=bib_bytes
    )

    # ENV for preprocess_bib_file
    env = {
        # Preprocessor config
        "UNIGRAM_NORMALIZER": "porter",

        # BIB INPUT S3
        "bib_file_S3_HOST": "http://127.0.0.1",
        "bib_file_S3_PORT": "9000",
        "bib_file_S3_ACCESS_KEY": MINIO_USER,
        "bib_file_S3_SECRET_KEY": MINIO_PWD,
        "bib_file_BUCKET_NAME": BUCKET_NAME,
        "bib_file_FILE_PATH": "",
        "bib_file_FILE_NAME": input_file_name,
        "bib_file_SELECTED_ATTRIBUTE": "abstract",

        # PostgreSQL output
        "normalized_docs_PG_HOST": "localhost",
        "normalized_docs_PG_PORT": "5432",
        "normalized_docs_PG_USER": PG_USER,
        "normalized_docs_PG_PASS": PG_PASS,
        "normalized_docs_DB_TABLE": "normalized_docs_bib",
    }

    for k, v in env.items():
        os.environ[k] = v

    # Run block
    preprocess_bib_file()

    # Query PostgreSQL for inserted documents
    engine = create_engine(
        f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@localhost:5432/"
    )
    df = pd.read_sql_table("normalized_docs_bib", engine)

    # Assertions
    assert len(df) > 0
    assert "doc_id" in df.columns
    assert "tokens" in df.columns

    # doc_id increments
    assert len(df["doc_id"]) == len(df)  # doc_id count matches rows
    assert df["doc_id"].is_unique         # no duplicates
    assert all(isinstance(x, str) for x in df["doc_id"])  # Bib IDs are strings

    assert set(df["doc_id"]) == {
        "WOS:001016714700004",
        "WOS:001322577100012"
    }

    df["tokens"] = df["tokens"].apply(parse_pg_array)

    assert isinstance(df.iloc[0]["tokens"], list)
    assert all(isinstance(t, str) for t in df.iloc[0]["tokens"])


def test_full_txt(s3_minio):
    input_file_name = "input"

    txt_path = Path(__file__).parent / "files" / f"{input_file_name}.txt"
    txt_bytes = txt_path.read_bytes()

    # Upload input to MinIO
    s3_minio.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{input_file_name}.txt",
        Body=txt_bytes
    )

    env = {
        "UNIGRAM_NORMALIZER": "porter",

        # TXT input S3
        "txt_file_S3_HOST": "http://127.0.0.1",
        "txt_file_S3_PORT": "9000",
        "txt_file_S3_ACCESS_KEY": MINIO_USER,
        "txt_file_S3_SECRET_KEY": MINIO_PWD,
        "txt_file_BUCKET_NAME": BUCKET_NAME,
        "txt_file_FILE_PATH": "",
        "txt_file_FILE_NAME": input_file_name,

        # Postgres output
        "normalized_docs_PG_HOST": "localhost",
        "normalized_docs_PG_PORT": "5432",
        "normalized_docs_PG_USER": PG_USER,
        "normalized_docs_PG_PASS": PG_PASS,
        "normalized_docs_DB_TABLE": "normalized_docs_txt",
    }

    for k, v in env.items():
        os.environ[k] = v

    preprocess_txt_file()

    # Query PostgreSQL
    engine = create_engine(
        f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@localhost:5432/"
    )
    df = pd.read_sql_table("normalized_docs_txt", engine)

    # Assertions
    assert len(df) > 0
    assert "doc_id" in df.columns
    assert "tokens" in df.columns
    assert len(df["doc_id"]) == len(df)

    df["tokens"] = df["tokens"].apply(parse_pg_array)

    assert isinstance(df.iloc[0]["tokens"], list)
    assert all(isinstance(t, str) for t in df.iloc[0]["tokens"])
