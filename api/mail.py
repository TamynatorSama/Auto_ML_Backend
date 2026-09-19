"""
mail.py
-------
Email over SMTP with STARTTLS, set in .env: SMTP_HOST, SMTP_PORT (587),
SMTP_USER, SMTP_PASSWORD and MAIL_FROM (default SMTP_USER). A Gmail app
password (smtp.gmail.com) is enough for a few testers; Resend, Postmark or SES
swap in by changing those values. With no SMTP_HOST the email goes to the
API's log instead.
"""

import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger("api.mail")

TEXTS = {
    "verify": ("Confirm your email for AutoML",
               "Confirm your email to finish signing up:\n\n{link}\n\n"
               "The link works for 24 hours. If you didn't sign up, ignore this email."),
    "reset": ("Reset your AutoML password",
              "Set a new password here:\n\n{link}\n\n"
              "The link works once, for an hour. If you didn't ask for it, ignore this email: "
              "your password stays as it is."),
}


def send_link(to: str, name: str, purpose: str, link: str) -> None:
    subject, body = TEXTS[purpose]
    send(to, subject, f"Hi {name},\n\n" + body.format(link=link))


def send(to: str, subject: str, body: str) -> None:
    host = os.environ.get("SMTP_HOST")
    if not host:
        log.warning("SMTP_HOST is not set; the email to %s says:\n%s", to, body)
        return
    message = EmailMessage()
    message["From"] = os.environ.get("MAIL_FROM") or os.environ["SMTP_USER"]
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT") or 587), timeout=30) as smtp:
            smtp.starttls()
            smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException):
        log.exception("could not send the email to %s", to)
