"""Trial outcome labels used by every log record and report (IP-9, §3.2)."""

from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    """Spec outcome strings. Member names are identifiers; values are verbatim spec text."""

    BLOCKED_INGRESS = "Blocked-ingress"
    ADMITTED_REFUSED = "Admitted-refused"
    EGRESS_SAVED = "Egress-saved"
    SUCCESS = "Success"
    ERROR = "Error"
