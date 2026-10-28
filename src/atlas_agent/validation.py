"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    errors = []
    args = decision.arguments

    if "user_id" in args:
        if args["user_id"] != request.authenticated_user_id:
            errors.append("user_id_must_match_authenticated_user")

    for field in ("project_id", "resource_id", "start", "end"):
        trusted_value = getattr(request, field)

        if decision.tool == "reserve_resource" and trusted_value is None:
            errors.append(f"{field}_unresolved_in_request")

        elif trusted_value is not None and field in args:
            if args[field] != trusted_value:
                errors.append(f"{field}_does_not_match_request")

    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    errors = []

    error_codes = {
        "arguments_complete": "reservation_arguments_incomplete",
        "policy_retrieved": "policy_not_retrieved",
        "authorizations_checked": "authorizations_not_checked",
        "resource_status_checked": "resource_status_not_checked",
        "project_authorized": "project_not_authorized",
        "qualification_current": "required_qualification_missing",
        "resource_available": "resource_not_available",
        "calibration_current": "calibration_not_current",
        "maintenance_clear": "maintenance_not_clear",
        "after_hours_approval_satisfied": "after_hours_approval_missing",
    }

    for field, error_code in error_codes.items():
        if not getattr(context, field):
            errors.append(error_code)

    return errors


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    errors = []
    valid_citations = retrieved_passage_ids(trace)
    invalid_citations = [
        citation for citation in response.citations if citation not in valid_citations
    ]
    if invalid_citations:
        errors.append("citations_not_retrieved: " + ", ".join(invalid_citations))
    successful_actions = {
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok
    }
    for action in response.completed_actions:
        if action not in successful_actions:
            errors.append("completed_action_not_in_successful_trace")
    if response.status == "completed" and not successful_actions:
        errors.append("completed_without_successful_mutation")
    for action in successful_actions:
        if action not in response.completed_actions:
            errors.append("successful_mutation_not_reported")
    if response.status != "completed" and response.completed_actions:
        errors.append("non_completed_response_claims_actions")

    return errors
