"""Optional AWS path: train on SageMaker, store models and reviews in S3, serve on ECS Fargate.

Every function takes boto3 clients as arguments, so the control UI, scripts and tests share one code path
(tests pass fakes; no AWS account needed to run them). Infrastructure is in infra/terraform.

Flow:
    upload_dataset -> launch_training (SageMaker, GPU, optional spot) -> job_status
    -> promote (copy the job's ONNX folder to s3://<bucket>/models/active/) -> restart_service (ECS picks it up)
"""

import io
import json
import os
import tarfile
import time
from datetime import datetime, timezone

PREFIX = "spendcoder"
DATA_KEY = "data/spend_lines.csv.gz"
ACTIVE_PREFIX = "models/active/onnx/"
ONNX_FILES = ("model.onnx", "tokenizer.json", "spendcoder.json")
# A Hugging Face PyTorch training image (AWS Deep Learning Containers). Check the current list in the
# "Available Deep Learning Containers Images" page and override with SAGEMAKER_IMAGE_URI if needed.
DEFAULT_IMAGE = ("763104351884.dkr.ecr.{region}.amazonaws.com/"
                 "huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04")
SAGEMAKER_REQUIREMENTS = """transformers>=4.46
datasets>=2.19
accelerate>=0.30
onnxscript>=0.1
onnxruntime>=1.18
tokenizers>=0.19
huggingface_hub>=0.24
pandas>=2.0
scikit-learn>=1.4
requests>=2.31
"""


def package_source(repo_root):
    """The code SageMaker runs: spendcoder/ and scripts/, plus a training-only requirements.txt."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ("spendcoder", "scripts"):
            for root, _, files in os.walk(os.path.join(repo_root, folder)):
                for name in files:
                    if name.endswith(".py"):
                        full = os.path.join(root, name)
                        tar.add(full, arcname=os.path.relpath(full, repo_root))
        req = SAGEMAKER_REQUIREMENTS.encode()
        info = tarfile.TarInfo("requirements.txt")
        info.size = len(req)
        tar.addfile(info, io.BytesIO(req))
    return buf.getvalue()


def upload_dataset(s3, bucket, local_path, key=DATA_KEY):
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def launch_training(s3, sm, bucket, role_arn, repo_root, region, image_uri=None, data_key=DATA_KEY,
                    instance_type="ml.g4dn.xlarge", epochs=4, spot=True, max_hours=2, job_name=None):
    """Start a SageMaker training job that runs scripts/train.py on a GPU. Returns the job name."""
    job_name = job_name or f"{PREFIX}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    code_key = f"code/{job_name}/sourcedir.tar.gz"
    s3.put_object(Bucket=bucket, Key=code_key, Body=package_source(repo_root))
    hp = {   # the training toolkit passes these to the script as --name value (values JSON-encoded)
        "sagemaker_program": "scripts/train.py",
        "sagemaker_submit_directory": f"s3://{bucket}/{code_key}",
        "sagemaker_region": region,
        "data": f"/opt/ml/input/data/train/{os.path.basename(data_key)}",
        "out": "/opt/ml/model",
        "epochs": epochs,
    }
    stop = {"MaxRuntimeInSeconds": int(max_hours * 3600)}
    if spot:
        stop["MaxWaitTimeInSeconds"] = int(max_hours * 3600 * 2)
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={"TrainingImage": image_uri or DEFAULT_IMAGE.format(region=region),
                                "TrainingInputMode": "File"},
        HyperParameters={k: json.dumps(v) for k, v in hp.items()},
        InputDataConfig=[{"ChannelName": "train", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/{data_key}", "S3DataDistributionType": "FullyReplicated"}}}],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/models/jobs/"},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition=stop,
        EnableManagedSpotTraining=bool(spot),
        Tags=[{"Key": "project", "Value": PREFIX}],
    )
    return job_name


def job_status(sm, job_name):
    d = sm.describe_training_job(TrainingJobName=job_name)
    return {"name": job_name, "status": d["TrainingJobStatus"], "detail": d.get("SecondaryStatus"),
            "started": str(d.get("TrainingStartTime") or d.get("CreationTime")),
            "model_uri": d.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
            "billable_seconds": d.get("BillableTimeInSeconds"), "failure": d.get("FailureReason")}


def list_jobs(sm, max_results=10):
    resp = sm.list_training_jobs(NameContains=PREFIX, SortBy="CreationTime", SortOrder="Descending",
                                 MaxResults=max_results)
    return [{"name": j["TrainingJobName"], "status": j["TrainingJobStatus"], "created": str(j["CreationTime"])}
            for j in resp.get("TrainingJobSummaries", [])]


def _split_s3(uri):
    bucket, _, key = uri.replace("s3://", "", 1).partition("/")
    return bucket, key


def promote(s3, sm, bucket, job_name, promoted_by="control UI"):
    """Copy a finished job's ONNX folder to models/active/ and record what is in production."""
    status = job_status(sm, job_name)
    if status["status"] != "Completed":
        raise ValueError(f"Job {job_name} is {status['status']}, not Completed")
    src_bucket, key = _split_s3(status["model_uri"])
    body = s3.get_object(Bucket=src_bucket, Key=key)["Body"].read()
    found = {}
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        for m in tar.getmembers():
            name = os.path.basename(m.name)
            if m.isfile() and "onnx" in m.name.split("/")[:-1] and name in ONNX_FILES:
                found[name] = tar.extractfile(m).read()
    missing = [f for f in ONNX_FILES if f not in found]
    if missing:
        raise ValueError(f"Job {job_name} has no ONNX export (missing {missing})")
    for name, data in found.items():
        s3.put_object(Bucket=bucket, Key=ACTIVE_PREFIX + name, Body=data)
    record = {"job": job_name, "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "promoted_by": promoted_by, "metrics": json.loads(found["spendcoder.json"]).get("metrics", {})}
    s3.put_object(Bucket=bucket, Key="models/active/active.json", Body=json.dumps(record, indent=2).encode())
    return record


def active_model(s3, bucket):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key="models/active/active.json")["Body"].read())
    except Exception:
        return None


def download_active(s3, bucket, local_dir):
    """Used by the API at start-up when MODEL_S3_URI points to the active folder."""
    os.makedirs(local_dir, exist_ok=True)
    for name in ONNX_FILES:
        s3.download_file(bucket, ACTIVE_PREFIX + name, os.path.join(local_dir, name))
    return local_dir


def restart_service(ecs, cluster, service):
    """Roll the ECS service so new tasks load the newly promoted model."""
    ecs.update_service(cluster=cluster, service=service, forceNewDeployment=True)
    return f"Restarting {service} in {cluster}"


def save_reviews(s3, bucket, csv_bytes):
    """People's corrections become training data for the next round."""
    key = f"reviews/{datetime.now(timezone.utc):%Y/%m/%d/%H%M%S}.csv"
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    return f"s3://{bucket}/{key}"


def wait(sm, job_name, poll=60, log=print):
    while True:
        s = job_status(sm, job_name)
        log(f"{s['status']} - {s['detail']}")
        if s["status"] in ("Completed", "Failed", "Stopped"):
            return s
        time.sleep(poll)
