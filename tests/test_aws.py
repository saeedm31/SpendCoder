"""AWS path with fake clients: no account, no network."""

import io
import json
import os
import tarfile
import unittest

from spendcoder import aws

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Body:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": Body(self.objects[(Bucket, Key)])}

    def upload_file(self, path, bucket, key):
        self.objects[(bucket, key)] = open(path, "rb").read()

    def download_file(self, bucket, key, path):
        open(path, "wb").write(self.objects[(bucket, key)])


class FakeSageMaker:
    def __init__(self, status="Completed"):
        self.created, self.status = [], status

    def create_training_job(self, **kw):
        self.created.append(kw)

    def describe_training_job(self, TrainingJobName):
        return {"TrainingJobStatus": self.status, "SecondaryStatus": "Done", "CreationTime": "t",
                "ModelArtifacts": {"S3ModelArtifacts": "s3://b/models/jobs/j/output/model.tar.gz"}}

    def list_training_jobs(self, **kw):
        return {"TrainingJobSummaries": [{"TrainingJobName": "spendcoder-1", "TrainingJobStatus": "Completed",
                                          "CreationTime": "t"}]}


def model_tar(with_onnx=True):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        files = {"config.json": b"{}"}
        if with_onnx:
            files.update({"onnx/model.onnx": b"onnx", "onnx/tokenizer.json": b"{}",
                          "onnx/spendcoder.json": json.dumps({"metrics": {"finetuned_test": {"macro_f1": 0.8}}}).encode()})
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class AwsTests(unittest.TestCase):
    def test_source_package_has_code_and_training_requirements(self):
        with tarfile.open(fileobj=io.BytesIO(aws.package_source(ROOT)), mode="r:gz") as tar:
            names = tar.getnames()
        self.assertIn("scripts/train.py", names)
        self.assertIn("spendcoder/finetune.py", names)
        self.assertIn("requirements.txt", names)
        self.assertFalse(any(n.startswith(("tests", "serve")) for n in names))

    def test_launch_training_job(self):
        s3, sm = FakeS3(), FakeSageMaker()
        name = aws.launch_training(s3, sm, "b", "arn:role", ROOT, "eu-west-1", epochs=3, spot=True, job_name="spendcoder-x")
        self.assertEqual(name, "spendcoder-x")
        job = sm.created[0]
        hp = {k: json.loads(v) for k, v in job["HyperParameters"].items()}
        self.assertEqual(hp["sagemaker_program"], "scripts/train.py")
        self.assertEqual(hp["epochs"], 3)
        self.assertEqual(hp["out"], "/opt/ml/model")
        self.assertTrue(job["EnableManagedSpotTraining"])
        self.assertGreater(job["StoppingCondition"]["MaxWaitTimeInSeconds"], job["StoppingCondition"]["MaxRuntimeInSeconds"])
        self.assertIn(("b", "code/spendcoder-x/sourcedir.tar.gz"), s3.objects)
        self.assertIn("eu-west-1", job["AlgorithmSpecification"]["TrainingImage"])

    def test_promote_copies_onnx_and_records_it(self):
        s3, sm = FakeS3(), FakeSageMaker()
        s3.objects[("b", "models/jobs/j/output/model.tar.gz")] = model_tar()
        record = aws.promote(s3, sm, "b", "spendcoder-1")
        self.assertEqual(record["metrics"]["finetuned_test"]["macro_f1"], 0.8)
        for f in aws.ONNX_FILES:
            self.assertIn(("b", aws.ACTIVE_PREFIX + f), s3.objects)
        self.assertEqual(aws.active_model(s3, "b")["job"], "spendcoder-1")

    def test_promote_refuses_unfinished_or_incomplete_jobs(self):
        with self.assertRaises(ValueError):
            aws.promote(FakeS3(), FakeSageMaker(status="InProgress"), "b", "j")
        s3 = FakeS3()
        s3.objects[("b", "models/jobs/j/output/model.tar.gz")] = model_tar(with_onnx=False)
        with self.assertRaises(ValueError):
            aws.promote(s3, FakeSageMaker(), "b", "j")

    def test_jobs_reviews_restart(self):
        self.assertEqual(aws.list_jobs(FakeSageMaker())[0]["name"], "spendcoder-1")
        s3 = FakeS3()
        uri = aws.save_reviews(s3, "b", b"text,category\n")
        self.assertTrue(uri.startswith("s3://b/reviews/"))
        self.assertIsNone(aws.active_model(s3, "b"))

        class Ecs:
            def update_service(self, **kw):
                self.kw = kw
        ecs = Ecs()
        aws.restart_service(ecs, "c", "s")
        self.assertTrue(ecs.kw["forceNewDeployment"])


if __name__ == "__main__":
    unittest.main()
