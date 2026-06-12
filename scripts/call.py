#!/usr/bin/env python3
"""
Make an outbound call with Twilio's REST API.

Usage examples:
  # from env vars
  python scripts/call.py --to +91XXXXXXXXXX

  # explicit args
  python scripts/call.py --sid ACxxx --token yourtoken --from +1TWILIO --to +91XXXXXXXXXX --url https://your-ngrok.ngrok-free.app/twilio/voice

Environment variables used (optional):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM (Twilio number), CALL_TO, PUBLIC_BASE_URL

Requires: `requests` (pip install requests)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.parse
import urllib.request


def make_call(account_sid: str, auth_token: str, from_number: str, to_number: str, url: str) -> tuple[int, str]:
    endpoint = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json"
    data = urllib.parse.urlencode({"To": to_number, "From": from_number, "Url": url}).encode()
    auth = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Authorization", f"Basic {auth}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return resp.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, 'read') else ''
        return e.code, body


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make outbound call via Twilio")
    p.add_argument("--to", help="Destination number in E.164 (e.g. +911234567890)")
    p.add_argument("--from", dest="from_", help="Twilio number in E.164")
    p.add_argument("--url", help="TwiML URL to use for the call (default PUBLIC_BASE_URL/twilio/voice)")
    p.add_argument("--sid", help="Twilio Account SID")
    p.add_argument("--token", help="Twilio Auth Token")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    account_sid = args.sid or os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = args.token or os.getenv("TWILIO_AUTH_TOKEN")
    to_number = args.to or os.getenv("CALL_TO") or os.getenv("outgoing_number") or os.getenv("OUTGOING_NUMBER")
    from_number = args.from_ or os.getenv("TWILIO_FROM") or os.getenv("TWILIO_NUMBER")

    public_base = os.getenv("PUBLIC_BASE_URL")
    url = args.url
    if not url and public_base and to_number:
        url = public_base.rstrip("/") + f"/twilio/voice?to_number={urllib.parse.quote(to_number)}"
    elif not url and public_base:
        url = public_base.rstrip("/") + "/twilio/voice"

    missing = [name for name, val in (
        ("TWILIO_ACCOUNT_SID", account_sid),
        ("TWILIO_AUTH_TOKEN", auth_token),
        ("To number (--to or CALL_TO)", to_number),
        ("From number (--from or TWILIO_FROM/TWILIO_NUMBER)", from_number),
        ("TwiML URL (--url or PUBLIC_BASE_URL)", url),
    )
    if not val]

    if missing:
        print("Missing required parameters:")
        for m in missing:
            print(f" - {m}")
        print("\nProvide via command-line arguments or environment variables.")
        sys.exit(2)

    status, body = make_call(account_sid, auth_token, from_number, to_number, url)
    print(f"HTTP {status}")
    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2))
    except Exception:
        print(body)

    if status >= 400:
        sys.exit(1)


if __name__ == "__main__":
    main()
