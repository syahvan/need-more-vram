#!/usr/bin/env python3
"""Build the site image and deploy it to Cloud Run without the gcloud CLI.

Uses a service-account key and the public REST APIs (Service Usage, Artifact
Registry, Cloud Run Admin v2). Docker is used for build/push.

    GOOGLE_APPLICATION_CREDENTIALS=key.json python3 deploy/cloudrun.py \
        --region asia-southeast2 --service need-more-vram [--docker "sudo docker"]

Requires: google-auth, requests (both ship with most Python setups).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import google.auth.transport.requests
import requests
from google.oauth2 import service_account

ROOT = Path(__file__).resolve().parent.parent
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
APIS = ["run.googleapis.com", "artifactregistry.googleapis.com"]


def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


class Gcp:
    def __init__(self, key_path: str):
        self.creds = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)
        self.project = self.creds.project_id
        self.session = requests.Session()

    def token(self) -> str:
        if not self.creds.valid:
            self.creds.refresh(google.auth.transport.requests.Request())
        return self.creds.token

    def call(self, method: str, url: str, body: dict | None = None, ok=(200,)) -> dict:
        r = self.session.request(method, url, json=body, headers={"Authorization": f"Bearer {self.token()}"}, timeout=60)
        if r.status_code not in ok:
            raise RuntimeError(f"{method} {url} -> {r.status_code}: {r.text[:800]}")
        return r.json() if r.text else {}

    def wait(self, op_url: str, what: str, timeout: int = 600) -> dict:
        start = time.time()
        while True:
            op = self.call("GET", op_url)
            if op.get("done"):
                if "error" in op:
                    raise RuntimeError(f"{what} failed: {json.dumps(op['error'])[:800]}")
                return op
            if time.time() - start > timeout:
                raise TimeoutError(f"{what} still running after {timeout}s")
            time.sleep(3)


def ensure_apis(g: Gcp) -> None:
    for api in APIS:
        url = f"https://serviceusage.googleapis.com/v1/projects/{g.project}/services/{api}"
        try:
            state = g.call("GET", url).get("state")
        except RuntimeError as e:
            log(f"cannot read state of {api} ({e}); assuming enabled")
            continue
        if state == "ENABLED":
            continue
        log(f"enabling {api}")
        op = g.call("POST", f"{url}:enable", {})
        if op.get("name") and not op.get("done"):
            g.wait(f"https://serviceusage.googleapis.com/v1/{op['name']}", f"enable {api}")


def ensure_repo(g: Gcp, region: str, repo: str) -> None:
    base = f"https://artifactregistry.googleapis.com/v1/projects/{g.project}/locations/{region}/repositories"
    r = g.session.get(f"{base}/{repo}", headers={"Authorization": f"Bearer {g.token()}"}, timeout=60)
    if r.status_code == 200:
        return
    if r.status_code != 404:
        raise RuntimeError(f"reading repository: {r.status_code} {r.text[:500]}")
    log(f"creating Artifact Registry repo {repo} in {region}")
    op = g.call("POST", f"{base}?repositoryId={repo}", {"format": "DOCKER", "description": "need-more-vram images"})
    g.wait(f"https://artifactregistry.googleapis.com/v1/{op['name']}", "create repository")


def run(cmd: list[str], stdin: str | None = None) -> None:
    log("$ " + " ".join(shlex.quote(c) for c in cmd if "token" not in c.lower()))
    subprocess.run(cmd, input=stdin, text=True, check=True)


def build_and_push(g: Gcp, docker: list[str], image: str, registry: str) -> None:
    run(docker + ["login", "-u", "oauth2accesstoken", "--password-stdin", f"https://{registry}"], stdin=g.token())
    run(docker + ["build", "--pull", "-t", image, str(ROOT)])
    run(docker + ["push", image])


def deploy_service(g: Gcp, region: str, service: str, image: str, public: bool) -> str:
    parent = f"projects/{g.project}/locations/{region}"
    base = f"https://run.googleapis.com/v2/{parent}/services"
    spec = {
        "template": {
            "containers": [
                {
                    "image": image,
                    "ports": [{"containerPort": 8080}],
                    "resources": {"limits": {"cpu": "1", "memory": "256Mi"}, "cpuIdle": True},
                }
            ],
            "scaling": {"minInstanceCount": 0, "maxInstanceCount": 3},
            "maxInstanceRequestConcurrency": 80,
        },
        "ingress": "INGRESS_TRAFFIC_ALL",
    }
    exists = g.session.get(f"{base}/{service}", headers={"Authorization": f"Bearer {g.token()}"}, timeout=60).status_code == 200
    if exists:
        log(f"updating service {service}")
        op = g.call("PATCH", f"{base}/{service}", spec)
    else:
        log(f"creating service {service}")
        op = g.call("POST", f"{base}?serviceId={service}", spec)
    g.wait(f"https://run.googleapis.com/v2/{op['name']}", "deploy")

    if public:
        policy_url = f"{base}/{service}"
        policy = g.call("GET", f"{policy_url}:getIamPolicy")
        bindings = policy.get("bindings", [])
        invoker = next((b for b in bindings if b["role"] == "roles/run.invoker"), None)
        if not invoker:
            invoker = {"role": "roles/run.invoker", "members": []}
            bindings.append(invoker)
        if "allUsers" not in invoker["members"]:
            invoker["members"].append("allUsers")
            policy["bindings"] = bindings
            try:
                g.call("POST", f"{policy_url}:setIamPolicy", {"policy": policy})
                log("granted public (allUsers) invoker access")
            except RuntimeError as e:
                if "403" not in str(e):
                    raise
                log(
                    "deployed, but this service account lacks run.services.setIamPolicy, so the service is still "
                    "private. Ask a project admin to allow unauthenticated access (Cloud Run > service > Security), "
                    "or grant the service account roles/run.admin and rerun."
                )

    return g.call("GET", f"{base}/{service}").get("uri", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), help="service-account JSON key")
    ap.add_argument("--region", default="asia-southeast2")
    ap.add_argument("--service", default="need-more-vram")
    ap.add_argument("--repo", default="need-more-vram", help="Artifact Registry repository")
    ap.add_argument("--docker", default="docker", help='docker command, e.g. "sudo docker"')
    ap.add_argument("--private", action="store_true", help="do not grant allUsers invoker")
    ap.add_argument("--check", action="store_true", help="only verify credentials and API access")
    args = ap.parse_args()
    if not args.key:
        ap.error("pass --key or set GOOGLE_APPLICATION_CREDENTIALS")

    g = Gcp(args.key)
    log(f"project {g.project} as {g.creds.service_account_email}")
    ensure_apis(g)
    if args.check:
        g.call("GET", f"https://run.googleapis.com/v2/projects/{g.project}/locations/{args.region}/services")
        log("credentials can list Cloud Run services — OK")
        return 0

    ensure_repo(g, args.region, args.repo)
    registry = f"{args.region}-docker.pkg.dev"
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    image = f"{registry}/{g.project}/{args.repo}/{args.service}:{tag}"
    build_and_push(g, shlex.split(args.docker), image, registry)
    url = deploy_service(g, args.region, args.service, image, public=not args.private)
    log(f"live at {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
