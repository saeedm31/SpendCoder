output "bucket" {
  value = aws_s3_bucket.main.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "api_url" {
  value = "http://${aws_lb.api.dns_name}"
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service" {
  value = aws_ecs_service.api.name
}

output "operator_policy_arn" {
  value = aws_iam_policy.operator.arn
}
