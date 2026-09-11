"""Outbound-only command bridge for the Camber notebook (no inbound networking).

Polls one ntfy topic for token-guarded JSON jobs and publishes results to another.
Deploy with environment variables:
  B1_CMD_TOPIC, B1_RES_TOPIC, B1_SECRET, optional B1_NTFY_BASE

This file is intentionally outside the B1 provenance paths (fine_tuning/, src/,
tests/) and lives on the ops-bridge branch only.
"""

import json
import os
import subprocess
import time

import requests

BASE = os.environ.get("B1_NTFY_BASE", "https://ntfy.sh")
CMD_TOPIC = os.environ["B1_CMD_TOPIC"]
RES_TOPIC = os.environ["B1_RES_TOPIC"]
SECRET = os.environ["B1_SECRET"]
MAX_STDOUT = int(os.environ.get("B1_MAX_STDOUT", "6000"))
MAX_STDERR = int(os.environ.get("B1_MAX_STDERR", "2000"))
POLL_SECONDS = float(os.environ.get("B1_POLL_SECONDS", "2"))


def publish(session, topic, payload):
    session.post(f"{BASE}/{topic}", data=json.dumps(payload), timeout=30)


def main():
    session = requests.Session()
    since = os.environ.get("B1_SINCE", "0")
    publish(session, RES_TOPIC, {"event": "agent_started", "pid": os.getpid()})
    while True:
        try:
            response = session.get(
                f"{BASE}/{CMD_TOPIC}/json",
                params={"poll": "1", "since": since},
                timeout=30,
            )
            if response.status_code == 200 and response.text.strip():
                for line in response.text.splitlines():
                    if not line.strip():
                        continue
                    message = json.loads(line)
                    if message.get("event") != "message":
                        continue
                    since = message.get("id", since)
                    try:
                        job = json.loads(message.get("message", ""))
                    except json.JSONDecodeError:
                        continue
                    if job.get("secret") != SECRET or "cmd" not in job:
                        continue
                    job_id = str(job.get("id", message.get("id")))
                    timeout = int(job.get("timeout", 1800))
                    try:
                        completed = subprocess.run(
                            ["bash", "-lc", job["cmd"]],
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        result = {
                            "id": job_id,
                            "exit": completed.returncode,
                            "stdout": completed.stdout[-MAX_STDOUT:],
                            "stderr": completed.stderr[-MAX_STDERR:],
                        }
                    except subprocess.TimeoutExpired:
                        result = {"id": job_id, "exit": -1, "stdout": "", "stderr": "timeout"}
                    publish(session, RES_TOPIC, result)
            time.sleep(POLL_SECONDS)
        except Exception:
            time.sleep(3)


if __name__ == "__main__":
    main()
