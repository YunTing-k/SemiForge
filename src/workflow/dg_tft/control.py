"""Explicit workflow termination shared by preparation and modeling agents.

Stopping without a valid model is distinct from accepting degraded quality.
The orchestrator owns persistence and reports; child agents only submit evidence.
"""
from __future__ import annotations


class WorkflowStopped(Exception):
    def __init__(self, reason: str, *, evidence: str = "", needs_input: bool = False,
                 required_information: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence
        self.needs_input = needs_input
        self.required_information = required_information

    def to_dict(self) -> dict:
        return {"reason": self.reason, "evidence": self.evidence,
                "needs_input": self.needs_input,
                "required_information": self.required_information}


def check_stop_decision(decision: dict) -> None:
    """Raise on an explicit stop; malformed stop requests also fail closed."""
    if decision.get("_tool") != "stop_workflow":
        return
    reason = str(decision.get("reason", "")).strip()
    evidence = str(decision.get("evidence", "")).strip()
    information = str(decision.get("required_information", "")).strip()
    needs_input = decision.get("outcome") == "needs_input"
    if decision.get("outcome") not in ("failed", "needs_input") or not reason or not evidence or (needs_input and not information):
        raise WorkflowStopped("Invalid stop_workflow decision: reason, evidence and any requested information are required.",
                              evidence=str(decision))
    raise WorkflowStopped(reason, evidence=evidence, needs_input=needs_input,
                          required_information=information)
