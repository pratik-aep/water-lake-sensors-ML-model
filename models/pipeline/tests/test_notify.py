import json

from models.pipeline import notify

REPORT = {
    "generated_at": "2025-08-21T23:50",
    "alerts": [
        {"severity": "medium", "station": "pichola", "message": "BOD likely above 3 mg/L"},
        {"severity": "low", "station": "pichola", "message": "Unusual readings"},
        {"severity": "high", "station": "goverdhan_sagar", "message": "Possible algal bloom"},
    ],
}


class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        self.user = user

    def send_message(self, message):
        FakeSMTP.sent.append(message)


def test_alerts_below_the_chosen_severity_are_left_out_and_the_worst_leads():
    chosen = notify.select(REPORT["alerts"], "medium")
    assert [a["severity"] for a in chosen] == ["high", "medium"]
    subject, body = notify.compose(REPORT, chosen)
    assert subject.startswith("[HIGH] 2 lake alert(s): goverdhan_sagar, pichola")
    assert "Unusual readings" not in body


def test_email_and_webhook_both_go_out_when_configured(monkeypatch):
    posted = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda request, timeout: posted.append(request) or Response())
    config = notify.config_from_env({"ALERT_EMAIL_TO": "a@x.org, b@x.org", "SMTP_HOST": "smtp.x.org", "SMTP_USER": "me@x.org",
                                     "SMTP_PASSWORD": "secret", "ALERT_WEBHOOK_URL": "https://hooks.example/abc"})
    sent = notify.notify(REPORT, config)
    assert sent == {"alerts": 2, "email": True, "webhook": True}
    assert FakeSMTP.sent[-1]["To"] == "a@x.org, b@x.org" and FakeSMTP.sent[-1]["From"] == "me@x.org"
    assert "Possible algal bloom" in json.loads(posted[0].data)["text"]


def test_a_quiet_day_sends_nothing(monkeypatch):
    monkeypatch.setattr(notify.smtplib, "SMTP", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no email expected")))
    config = notify.config_from_env({"ALERT_EMAIL_TO": "a@x.org", "SMTP_HOST": "smtp.x.org"})
    assert notify.notify({"generated_at": "t", "alerts": [REPORT["alerts"][1]]}, config) == {"alerts": 0, "email": False, "webhook": False}
