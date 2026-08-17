"""Versioned service and saturation boundaries derived from sealed M1 evidence."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from .kv_capacity_planner import StrictFrozenModel
from .m1_capacity_analysis import (
    CapacityKneePolicy,
    CapacityPointAnalysis,
    CapacitySweepAnalysis,
)

CAPACITY_BOUNDARY_VERSION: Literal["longctx-m1-capacity-boundaries.v2"] = (
    "longctx-m1-capacity-boundaries.v2"
)

BoundaryStatus = Literal[
    "bracketed",
    "left-censored-below-lowest-load",
    "right-censored-above-highest-load",
    "unresolved",
]


class CapacityBoundaryPointReference(StrictFrozenModel):
    """Compact reference to one fully preserved v1 capacity point."""

    load_id: str
    target_offered_requests_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    empirical_offered_requests_per_second: Optional[float] = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    classification: Literal["stable", "overloaded", "transitional", "invalid"]


class SLOServiceBoundary(StrictFrozenModel):
    """Highest stable load and first SLO/goodput breach, including censoring."""

    status: BoundaryStatus
    last_stable: Optional[CapacityBoundaryPointReference]
    first_slo_goodput_breach: Optional[CapacityBoundaryPointReference]
    accepted: bool
    failure_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_status_shape(self) -> "SLOServiceBoundary":
        if self.status == "bracketed" and (
            self.last_stable is None or self.first_slo_goodput_breach is None
        ):
            raise ValueError("bracketed SLO service boundary requires both bounds")
        if self.status == "left-censored-below-lowest-load" and (
            self.last_stable is not None or self.first_slo_goodput_breach is None
        ):
            raise ValueError("left-censored SLO service boundary requires only an upper bound")
        if self.status == "right-censored-above-highest-load" and (
            self.last_stable is None or self.first_slo_goodput_breach is not None
        ):
            raise ValueError("right-censored SLO service boundary requires only a lower bound")
        if self.accepted != (self.status != "unresolved" and not self.failure_reasons):
            raise ValueError("SLO service boundary acceptance disagrees with its status")
        return self


class JointSaturationBoundary(StrictFrozenModel):
    """First repeat-supported joint overload and its immediately preceding point."""

    status: BoundaryStatus
    last_pre_saturation: Optional[CapacityBoundaryPointReference]
    first_joint_overload: Optional[CapacityBoundaryPointReference]
    accepted: bool
    failure_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_status_shape(self) -> "JointSaturationBoundary":
        if self.status == "bracketed" and (
            self.last_pre_saturation is None or self.first_joint_overload is None
        ):
            raise ValueError("bracketed joint saturation boundary requires both bounds")
        if self.status == "right-censored-above-highest-load" and (
            self.last_pre_saturation is None or self.first_joint_overload is not None
        ):
            raise ValueError("right-censored saturation boundary requires only a lower bound")
        if self.status == "left-censored-below-lowest-load":
            raise ValueError("joint saturation cannot be left-censored without a paired load")
        if self.accepted != (self.status != "unresolved" and not self.failure_reasons):
            raise ValueError("joint saturation boundary acceptance disagrees with its status")
        return self


class ContextCapacityBoundaries(StrictFrozenModel):
    """Two distinct production and mechanism boundaries for one context."""

    context_id: str
    context_tokens: int = Field(gt=0)
    all_points_eligible: bool
    slo_service_boundary: SLOServiceBoundary
    joint_saturation_boundary: JointSaturationBoundary
    accepted: bool
    failure_reasons: tuple[str, ...]


class CapacityBoundaryAnalysis(StrictFrozenModel):
    """Fail-closed v2 interpretation of an unchanged v1 capacity sweep."""

    schema_version: Literal["longctx-m1-capacity-boundaries.v2"] = CAPACITY_BOUNDARY_VERSION
    source_schema_version: Literal["longctx-m1-capacity.v1"]
    source_v1_passed: bool
    source_v1_failure_reasons: tuple[str, ...]
    threshold_policy: CapacityKneePolicy
    numeric_thresholds_modified: Literal[False] = False
    service_boundary_rule: Literal["highest-stable-before-first-slo-goodput-breach"] = (
        "highest-stable-before-first-slo-goodput-breach"
    )
    saturation_boundary_rule: Literal["immediate-eligible-point-before-first-joint-overload"] = (
        "immediate-eligible-point-before-first-joint-overload"
    )
    below_lowest_result: Literal["left-censored-below-lowest-load"] = (
        "left-censored-below-lowest-load"
    )
    no_breach_or_overload_result: Literal["right-censored-above-highest-load"] = (
        "right-censored-above-highest-load"
    )
    contexts: tuple[ContextCapacityBoundaries, ...]
    accepted: bool
    failure_reasons: tuple[str, ...]


def _point_reference(point: CapacityPointAnalysis) -> CapacityBoundaryPointReference:
    empirical = (
        point.metrics.empirical_offered_requests_per_second.median
        if point.metrics is not None
        else None
    )
    return CapacityBoundaryPointReference(
        load_id=point.load_id,
        target_offered_requests_per_second=(point.target_offered_requests_per_second.median),
        empirical_offered_requests_per_second=empirical,
        classification=point.classification,
    )


def _service_boundary(points: tuple[CapacityPointAnalysis, ...]) -> SLOServiceBoundary:
    if not points or any(not point.eligible for point in points):
        return SLOServiceBoundary(
            status="unresolved",
            last_stable=None,
            first_slo_goodput_breach=None,
            accepted=False,
            failure_reasons=("capacity_points_are_missing_or_ineligible",),
        )

    breach_indices = [
        index for index, point in enumerate(points) if point.signals.slo_goodput_breach
    ]
    if not breach_indices:
        stable_indices = [
            index for index, point in enumerate(points) if point.classification == "stable"
        ]
        if stable_indices and len(stable_indices) == len(points):
            return SLOServiceBoundary(
                status="right-censored-above-highest-load",
                last_stable=_point_reference(points[stable_indices[-1]]),
                first_slo_goodput_breach=None,
                accepted=True,
                failure_reasons=(),
            )
        return SLOServiceBoundary(
            status="unresolved",
            last_stable=(_point_reference(points[stable_indices[-1]]) if stable_indices else None),
            first_slo_goodput_breach=None,
            accepted=False,
            failure_reasons=("no_slo_breach_but_not_all_points_are_stable",),
        )

    first_breach = breach_indices[0]
    if any(
        not points[index].signals.slo_goodput_breach for index in range(first_breach, len(points))
    ):
        return SLOServiceBoundary(
            status="unresolved",
            last_stable=None,
            first_slo_goodput_breach=_point_reference(points[first_breach]),
            accepted=False,
            failure_reasons=("slo_goodput_breach_is_not_monotonic",),
        )
    if any(point.classification == "stable" for point in points[first_breach + 1 :]):
        return SLOServiceBoundary(
            status="unresolved",
            last_stable=None,
            first_slo_goodput_breach=_point_reference(points[first_breach]),
            accepted=False,
            failure_reasons=("stable_point_after_first_slo_goodput_breach",),
        )
    if first_breach == 0:
        return SLOServiceBoundary(
            status="left-censored-below-lowest-load",
            last_stable=None,
            first_slo_goodput_breach=_point_reference(points[0]),
            accepted=True,
            failure_reasons=(),
        )

    stable_indices = [
        index
        for index, point in enumerate(points[:first_breach])
        if point.classification == "stable"
    ]
    if not stable_indices:
        return SLOServiceBoundary(
            status="unresolved",
            last_stable=None,
            first_slo_goodput_breach=_point_reference(points[first_breach]),
            accepted=False,
            failure_reasons=("no_stable_point_before_first_slo_goodput_breach",),
        )
    return SLOServiceBoundary(
        status="bracketed",
        last_stable=_point_reference(points[stable_indices[-1]]),
        first_slo_goodput_breach=_point_reference(points[first_breach]),
        accepted=True,
        failure_reasons=(),
    )


def _saturation_boundary(points: tuple[CapacityPointAnalysis, ...]) -> JointSaturationBoundary:
    if not points or any(not point.eligible for point in points):
        return JointSaturationBoundary(
            status="unresolved",
            last_pre_saturation=None,
            first_joint_overload=None,
            accepted=False,
            failure_reasons=("capacity_points_are_missing_or_ineligible",),
        )

    overload_indices = [index for index, point in enumerate(points) if point.signals.joint_overload]
    if not overload_indices:
        return JointSaturationBoundary(
            status="right-censored-above-highest-load",
            last_pre_saturation=_point_reference(points[-1]),
            first_joint_overload=None,
            accepted=True,
            failure_reasons=(),
        )

    first_overload = overload_indices[0]
    if first_overload == 0:
        return JointSaturationBoundary(
            status="unresolved",
            last_pre_saturation=None,
            first_joint_overload=_point_reference(points[0]),
            accepted=False,
            failure_reasons=("first_load_cannot_form_a_paired_saturation_boundary",),
        )
    if any(point.classification == "stable" for point in points[first_overload + 1 :]):
        return JointSaturationBoundary(
            status="unresolved",
            last_pre_saturation=_point_reference(points[first_overload - 1]),
            first_joint_overload=_point_reference(points[first_overload]),
            accepted=False,
            failure_reasons=("stable_point_after_first_joint_overload",),
        )
    return JointSaturationBoundary(
        status="bracketed",
        last_pre_saturation=_point_reference(points[first_overload - 1]),
        first_joint_overload=_point_reference(points[first_overload]),
        accepted=True,
        failure_reasons=(),
    )


def derive_capacity_boundaries(source: CapacitySweepAnalysis) -> CapacityBoundaryAnalysis:
    """Separate SLO service capacity from repeat-supported physical saturation."""

    contexts: list[ContextCapacityBoundaries] = []
    for context in source.contexts:
        service = _service_boundary(context.points)
        saturation = _saturation_boundary(context.points)
        all_points_eligible = bool(context.points) and all(
            point.eligible for point in context.points
        )
        reasons: list[str] = []
        if not all_points_eligible:
            reasons.append("not_all_capacity_points_are_eligible")
        reasons.extend(f"slo_service:{reason}" for reason in service.failure_reasons)
        reasons.extend(f"joint_saturation:{reason}" for reason in saturation.failure_reasons)
        contexts.append(
            ContextCapacityBoundaries(
                context_id=context.context_id,
                context_tokens=context.context_tokens,
                all_points_eligible=all_points_eligible,
                slo_service_boundary=service,
                joint_saturation_boundary=saturation,
                accepted=all_points_eligible and service.accepted and saturation.accepted,
                failure_reasons=tuple(reasons),
            )
        )

    failures = [
        f"context_{context.context_id}:{reason}"
        for context in contexts
        for reason in context.failure_reasons
    ]
    if not contexts:
        failures.append("no_capacity_contexts")
    return CapacityBoundaryAnalysis(
        source_schema_version=source.schema_version,
        source_v1_passed=source.passed,
        source_v1_failure_reasons=source.failure_reasons,
        threshold_policy=source.policy,
        contexts=tuple(contexts),
        accepted=bool(contexts) and all(context.accepted for context in contexts),
        failure_reasons=tuple(failures),
    )


__all__ = [
    "CAPACITY_BOUNDARY_VERSION",
    "CapacityBoundaryAnalysis",
    "CapacityBoundaryPointReference",
    "ContextCapacityBoundaries",
    "JointSaturationBoundary",
    "SLOServiceBoundary",
    "derive_capacity_boundaries",
]
