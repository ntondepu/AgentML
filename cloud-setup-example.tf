# Example Minimal Cloud Setup (AWS EKS with Terraform)

# This is a minimal template. You must customize values for your environment.

provider "aws" {
  region = var.aws_region
}

resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = var.cluster_role_arn

  vpc_config {
    subnet_ids = var.subnet_ids
  }
}

variable "aws_region" {}
variable "cluster_name" {}
variable "cluster_role_arn" {}
variable "subnet_ids" { type = list(string) }

output "cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}
