"""Fine-tune on AWS SageMaker instead of Colab, then (optionally) promote the model to production.

    export AWS_REGION=eu-west-1 SPENDCODER_BUCKET=<bucket> SAGEMAKER_ROLE_ARN=<role arn>   # from terraform output
    python scripts/aws_train.py                         # upload data, train on ml.g4dn.xlarge (spot), wait
    python scripts/aws_train.py --promote               # ... and copy the ONNX model to models/active/
    python scripts/aws_train.py --promote --restart     # ... and roll the ECS API service to load it

AWS credentials come from your usual AWS profile or environment; nothing secret is stored in the repo.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spendcoder import aws  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/spend_lines.csv.gz")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--on-demand", action="store_true", help="do not use spot capacity")
    p.add_argument("--promote", action="store_true")
    p.add_argument("--restart", action="store_true")
    a = p.parse_args()

    import boto3

    region = os.environ.get("AWS_REGION", "eu-west-1")
    bucket, role = os.environ["SPENDCODER_BUCKET"], os.environ["SAGEMAKER_ROLE_ARN"]
    s3, sm = boto3.client("s3", region_name=region), boto3.client("sagemaker", region_name=region)

    print("Uploaded", aws.upload_dataset(s3, bucket, a.data))
    job = aws.launch_training(s3, sm, bucket, role, ROOT, region, image_uri=os.environ.get("SAGEMAKER_IMAGE_URI"),
                              instance_type=a.instance_type, epochs=a.epochs, spot=not a.on_demand)
    print("Started", job)
    result = aws.wait(sm, job)
    if result["status"] != "Completed":
        sys.exit(f"Training ended with {result['status']}: {result['failure']}")
    print("Model:", result["model_uri"])
    if a.promote:
        print("Promoted:", aws.promote(s3, sm, bucket, job, promoted_by="aws_train.py"))
    if a.restart:
        ecs = boto3.client("ecs", region_name=region)
        print(aws.restart_service(ecs, os.environ["ECS_CLUSTER"], os.environ["ECS_SERVICE"]))


if __name__ == "__main__":
    main()
