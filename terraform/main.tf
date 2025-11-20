terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0" # ajuste conforme versão
    }
  }
}

provider "google" {
  project = "SEU_PROJETO"
  region  = "us-central1"
}

# Criação do Cloud Run Job
resource "google_cloud_run_v2_job" "example_job" {
  name     = "meu-job"
  location = "us-central1"

  template {
    template {
      containers {
        image = "gcr.io/SEU_PROJETO/minha-imagem:latest"

        # Variáveis de ambiente (opcional)
        env {
          name  = "ENVIRONMENT"
          value = "production"
        }

        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }
      }

      max_retries   = 1
      timeout       = "3600s"
      service_account = "seu-service-account@SEU_PROJETO.iam.gserviceaccount.com"
    }
  }
}

# Scheduler que dispara o Job
resource "google_cloud_scheduler_job" "example_schedule" {
  name        = "meu-job-scheduler"
  description = "Executa o Cloud Run Job diariamente às 3h"
  schedule    = "0 3 * * *" # cron: todos os dias às 3h da manhã
  time_zone   = "America/Sao_Paulo"

  http_target {
    uri = "https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/SEU_PROJETO/jobs/meu-job:run"
    http_method = "POST"

    oauth_token {
      service_account_email = "seu-service-account@SEU_PROJETO.iam.gserviceaccount.com"
    }
  }
}
