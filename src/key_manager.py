"""Secret/key management helpers.

Provides a small abstraction to load the Gemini API key from a secure
source. By default it reads `GROQ_API_KEY` from the environment, or
from a file pointed to by `GROQ_API_KEY_FILE` (useful for Docker/K8s
secrets mounted as files). This module is intentionally lightweight and
does not require cloud SDKs; hooks show where to add a secrets manager.
"""
import os
from typing import Optional
import json

# Optional imports for cloud providers — imported lazily
_HAS_BOTO3 = False
_HAS_GCP = False
try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    boto3 = None

try:
    from google.cloud import secretmanager as gcp_secretmanager
    _HAS_GCP = True
except Exception:
    gcp_secretmanager = None


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def get_api_key() -> str:
    """Return the Gemini API key from a secure source.

    Order of resolution:
      1. `GROQ_API_KEY` env var
      2. File path in `GROQ_API_KEY_FILE` env var
      3. Empty string if not found

    Extend this function to integrate with AWS Secrets Manager, GCP Secret
    Manager, or your platform's secrets facility.
    """
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key

    file_path = os.environ.get("GROQ_API_KEY_FILE")
    if file_path:
        key = _read_file(file_path)
        if key:
            return key

    # Cloud provider integration
    provider = os.environ.get("SECRETS_PROVIDER", "")
    secret_name = os.environ.get("GROQ_SECRET_NAME")
    if provider.lower() == "aws" and _HAS_BOTO3 and secret_name:
        try:
            client = boto3.client("secretsmanager")
            resp = client.get_secret_value(SecretId=secret_name)
            secret_string = resp.get("SecretString")
            if secret_string:
                # Support both raw string secrets and JSON objects
                try:
                    parsed = json.loads(secret_string)
                    return parsed.get("GROQ_API_KEY", "")
                except Exception:
                    return secret_string.strip()
        except Exception:
            pass

    if provider.lower() == "gcp" and _HAS_GCP and secret_name:
        try:
            client = gcp_secretmanager.SecretManagerServiceClient()
            name = f"{secret_name}/versions/latest"
            resp = client.access_secret_version(name=name)
            payload = resp.payload.data.decode("UTF-8")
            try:
                parsed = json.loads(payload)
                return parsed.get("GROQ_API_KEY", "")
            except Exception:
                return payload.strip()
        except Exception:
            pass

    return ""
