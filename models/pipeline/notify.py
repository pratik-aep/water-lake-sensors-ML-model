"""Send a report's alerts by email and/or to a chat webhook (Slack, Teams, Discord, or a WhatsApp/SMS gateway).

Configured by environment variables, so no password lives in the code or the repository:
  ALERT_MIN_SEVERITY   high, medium (default) or low
  ALERT_EMAIL_TO       comma-separated recipients; with SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD,
                       ALERT_EMAIL_FROM (defaults to SMTP_USER)
  ALERT_WEBHOOK_URL    receives {"text": ..., "content": ...} as JSON
"""

import argparse
import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage
from pathlib import Path

from .report import SEVERITY


def config_from_env(env=os.environ) -> dict:
    return {
        "min_severity": env.get("ALERT_MIN_SEVERITY", "medium"),
        "email_to": [a.strip() for a in env.get("ALERT_EMAIL_TO", "").split(",") if a.strip()],
        "smtp_host": env.get("SMTP_HOST"),
        "smtp_port": int(env.get("SMTP_PORT", "587")),
        "smtp_user": env.get("SMTP_USER"),
        "smtp_password": env.get("SMTP_PASSWORD"),
        "email_from": env.get("ALERT_EMAIL_FROM") or env.get("SMTP_USER"),
        "webhook_url": env.get("ALERT_WEBHOOK_URL"),
    }


def select(alerts: list, min_severity: str) -> list:
    """Alerts at or above the chosen severity, most severe first."""
    keep = SEVERITY[: SEVERITY.index(min_severity) + 1]
    return sorted((a for a in alerts if a["severity"] in keep), key=lambda a: SEVERITY.index(a["severity"]))


def compose(report: dict, alerts: list) -> tuple[str, str]:
    worst = alerts[0]["severity"].upper()
    lakes = sorted({a["station"] for a in alerts})
    subject = f"[{worst}] {len(alerts)} lake alert(s): {', '.join(lakes)} ({report['generated_at']})"
    lines = [f"Lake monitoring report {report['generated_at']}", ""]
    lines += [f"[{a['severity'].upper()}] {a['station']}: {a['message']}" for a in alerts]
    lines += ["", "Full report and dashboard are in the reports folder for this run."]
    return subject, "\n".join(lines)


def send_email(subject: str, body: str, config: dict) -> None:
    message = EmailMessage()
    message["Subject"], message["From"], message["To"] = subject, config["email_from"], ", ".join(config["email_to"])
    message.set_content(body)
    with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=30) as smtp:
        smtp.starttls()
        if config["smtp_user"]:
            smtp.login(config["smtp_user"], config["smtp_password"])
        smtp.send_message(message)


def send_webhook(text: str, url: str) -> None:
    # Slack and Teams read "text"; Discord reads "content".
    data = json.dumps({"text": text, "content": text}).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30):
        pass


def notify(report: dict, config: dict, dry_run: bool = False) -> dict:
    """Send what the report raised; returns which channels were used (nothing is sent when there are no alerts)."""
    alerts = select(report["alerts"], config["min_severity"])
    sent = {"alerts": len(alerts), "email": False, "webhook": False}
    if not alerts:
        return sent
    subject, body = compose(report, alerts)
    if dry_run:
        print(f"(dry run) {subject}\n{body}")
        return sent
    if config["email_to"] and config["smtp_host"]:
        send_email(subject, body, config)
        sent["email"] = True
    if config["webhook_url"]:
        send_webhook(f"{subject}\n{body}", config["webhook_url"])
        sent["webhook"] = True
    return sent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True, help="report.json written by run")
    parser.add_argument("--dry-run", action="store_true", help="print the message instead of sending it")
    args = parser.parse_args(argv)
    config = config_from_env()
    if not args.dry_run and not (config["email_to"] and config["smtp_host"]) and not config["webhook_url"]:
        parser.error("no channel configured: set ALERT_EMAIL_TO and SMTP_HOST, or ALERT_WEBHOOK_URL (or use --dry-run)")
    sent = notify(json.loads(args.report.read_text()), config, args.dry_run)
    print(f"{sent['alerts']} alert(s); email sent: {sent['email']}; webhook sent: {sent['webhook']}")


if __name__ == "__main__":
    main()
