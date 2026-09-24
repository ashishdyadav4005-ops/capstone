"""Script to tag and push Counterfactual Pricing project Docker containers to Docker Hub.

Usage:
    python scripts/push_docker_hub.py <your_dockerhub_username> [--version v1.0.0] [--build]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


SERVICES = [
    {
        "service": "api",
        "possible_local_tags": [
            "capstone_project-api:latest",
            "capstone_project-api",
            "bds39_pricing_api:latest",
            "bds39_pricing_api",
        ],
        "remote_name": "pricing-api",
    },
    {
        "service": "dashboard",
        "possible_local_tags": [
            "capstone_project-dashboard:latest",
            "capstone_project-dashboard",
            "bds39_pricing_dashboard:latest",
            "bds39_pricing_dashboard",
        ],
        "remote_name": "pricing-dashboard",
    },
    {
        "service": "mlflow",
        "possible_local_tags": [
            "capstone_project-mlflow:latest",
            "capstone_project-mlflow",
            "bds39_mlflow:latest",
            "bds39_mlflow",
        ],
        "remote_name": "pricing-mlflow",
    },
]


def run_cmd(cmd: list[str]) -> bool:
    """Execute shell command and stream stdout."""
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def find_local_image(possible_tags: list[str]) -> str | None:
    """Find which local tag exists in docker."""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    existing_images = [img.strip() for img in result.stdout.strip().splitlines() if img.strip()]

    for tag in possible_tags:
        clean_tag = tag if ":" in tag else f"{tag}:latest"
        if clean_tag in existing_images or tag in existing_images:
            return tag

    # Fallback partial matching
    for tag in possible_tags:
        base = tag.split(":")[0]
        for existing in existing_images:
            if existing.startswith(base):
                return existing

    return possible_tags[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag and push project Docker images to Docker Hub.")
    parser.add_argument("username", nargs="?", help="Your Docker Hub username")
    parser.add_argument("--version", default="v1.0.0", help="Release version tag (default: v1.0.0)")
    parser.add_argument("--build", action="store_true", help="Build images first before pushing")
    args = parser.parse_args()

    username = args.username
    if not username:
        try:
            username = input("Enter your Docker Hub username: ").strip()
        except EOFError:
            username = ""

    if not username:
        print("Error: Docker Hub username is required.")
        print("Usage: python scripts/push_docker_hub.py <your_dockerhub_username> [--version v1.0.0]")
        sys.exit(1)

    version = args.version

    print("=" * 70)
    print(" DYNAMIC PRICING ENGINE — DOCKER HUB PUBLISHING PIPELINE")
    print(f" - Docker Hub Namespace: {username}")
    print(f" - Release Version Tag:  {version}")
    print("=" * 70)

    # 1. Build images if requested
    if args.build:
        print("\nStep 1: Building project images via docker compose...")
        if not run_cmd(["docker", "compose", "build"]):
            print("Error: Docker compose build failed.")
            sys.exit(1)

    # 2. Tag and Push each service
    print("\nStep 2: Tagging and Pushing container images...")
    pushed_repos: list[str] = []

    for item in SERVICES:
        service = item["service"]
        local_tag = find_local_image(item["possible_local_tags"]) or item["possible_local_tags"][0]
        remote_repo = f"{username}/{item['remote_name']}"
        version_tag = f"{remote_repo}:{version}"
        latest_tag = f"{remote_repo}:latest"

        print(f"\n==================== [{service.upper()}] ====================")
        print(f"Local Image Source: {local_tag}")
        print(f"Target Tags:        {version_tag}  &  {latest_tag}")

        # Tag version
        if not run_cmd(["docker", "tag", local_tag, version_tag]):
            print(f"Warning: Could not tag {local_tag} as {version_tag}")
            continue

        # Tag latest
        run_cmd(["docker", "tag", local_tag, latest_tag])

        # Push version
        print(f"\nPushing {version_tag} to Docker Hub...")
        if run_cmd(["docker", "push", version_tag]):
            # Push latest
            print(f"Pushing {latest_tag} to Docker Hub...")
            run_cmd(["docker", "push", latest_tag])
            pushed_repos.append(remote_repo)
        else:
            print(f"\n❌ Push failed for {version_tag}.")
            print("Please ensure you are authenticated by running:")
            print("   docker login")
            print("and then re-run this script.")

    # 3. Summary
    print("\n" + "=" * 70)
    print(" DOCKER HUB PUBLISHING SUMMARY")
    print("=" * 70)
    if pushed_repos:
        print("✅ Successfully published the following images to Docker Hub:\n")
        for repo in pushed_repos:
            print(f"  📦 https://hub.docker.com/r/{repo}")
            print(f"     - docker pull {repo}:{version}")
            print(f"     - docker pull {repo}:latest\n")
        print("All team members and evaluators can now run the complete stack with:")
        print(f"  docker run -d -p 8000:8000 {username}/pricing-api:latest")
        print(f"  docker run -d -p 8501:8501 {username}/pricing-dashboard:latest")
        print(f"  docker run -d -p 5000:5000 {username}/pricing-mlflow:latest")
    else:
        print("⚠️ No images were pushed. Ensure you have run 'docker login' first.")
    print("=" * 70)


if __name__ == "__main__":
    main()
