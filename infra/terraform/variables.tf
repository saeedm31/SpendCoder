variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "project" {
  description = "Name prefix for all resources"
  type        = string
  default     = "spendcoder"
}

variable "allowed_cidrs" {
  description = "Who may call the API (for example your office IP as [\"203.0.113.10/32\"]). No default on purpose."
  type        = list(string)
}

variable "image_tag" {
  description = "Tag of the API image in ECR"
  type        = string
  default     = "latest"
}

variable "desired_count" {
  description = "Number of API tasks. Starts at 0: set 1 after the first model is promoted. 0 stops the cost."
  type        = number
  default     = 0
}
