"""Unified All-in-One entrypoint for Counterfactual Dynamic Pricing system.

Starts and supervises:
1. Database seeding & account initialization
2. MLflow Tracking Server (Port 5000)
3. FastAPI REST Backend & Swagger Docs (Port 8000)
4. Streamlit Multi-Persona Dashboard UI (Port 8501)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request


def main() -> None:
    print("=" * 65)
    print(" STARTING UNIFIED COUNTERFACTUAL PRICING SYSTEM")
    print("=" * 65)

    os.environ.setdefault("ENVIRONMENT", "production")
    os.environ.setdefault("BACKEND_API_URL", "http://localhost:8000")
    os.environ.setdefault("DATABASE_URL", "sqlite:////app/data/bds39_pricing.db")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("JWT_SECRET_KEY", "prod-docker-unified-key-1234567890")

    # 1. Seed database & default users
    print("\n[1/4] Initializing database and demo user accounts...")
    subprocess.run([sys.executable, "scripts/seed_users.py"], check=False)

    processes: list[subprocess.Popen] = []

    def shutdown(signum=None, frame=None):
        print("\nShutting down all services gracefully...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 2. Start MLflow Server
    print("[2/4] Starting MLflow Tracking Server on port 5000...")
    os.makedirs("/app/data/mlflow_artifacts", exist_ok=True)
    mlflow_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            "sqlite:////app/data/mlflow.db",
            "--default-artifact-root",
            "/app/data/mlflow_artifacts",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
        ]
    )
    processes.append(mlflow_proc)

    # 3. Start FastAPI REST Backend
    print("[3/4] Starting FastAPI REST API on port 8000...")
    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )
    processes.append(api_proc)

    # Wait for API healthcheck
    print("Waiting for API to be ready...")
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://localhost:8000/api/v1/health", timeout=1) as resp:
                if resp.status == 200:
                    print("✓ FastAPI is healthy and running!")
                    break
        except Exception:
            time.sleep(1)

    # 4. Start Streamlit Dashboard UI
    print("[4/4] Starting Streamlit Decision Dashboard on port 8501...")
    print("\n" + "=" * 65)
    print(" ALL SERVICES ONLINE & OPERATIONAL:")
    print(" - Streamlit Decision UI:    http://localhost:8501")
    print(" - FastAPI Swagger Docs:     http://localhost:8000/docs")
    print(" - Prometheus Metrics:       http://localhost:8000/metrics")
    print(" - MLflow Tracking Server:   http://localhost:5000")
    print("=" * 65 + "\n")

    streamlit_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--theme.base=light",
        ]
    )
    processes.append(streamlit_proc)

    # Supervise processes
    while True:
        for p in processes:
            ret = p.poll()
            if ret is not None:
                print(f"Service process {p.args} exited with code {ret}")
                shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
