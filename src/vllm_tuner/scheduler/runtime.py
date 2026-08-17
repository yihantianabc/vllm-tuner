"""Runtime integration for the workload-aware vLLM V1 Scheduler."""

from __future__ import annotations

import os
import time

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tuner.config.models import AdaptivePrefillConfig
from vllm_tuner.runtime.server import (
    SLOTUNE_SCHEDULER_CONFIG_ENV,
    SLOTUNE_SCHEDULER_LOG_ENV,
)

from .controller import AdaptivePrefillController, ControllerDecision
from .instrumentation import (
    SchedulerDecisionWriter,
    SchedulerSignals,
    SchedulerStepRecord,
    collect_scheduler_signals,
    has_unfinished_prefill,
    split_scheduled_tokens,
)

logger = init_logger(__name__)


class AdaptivePrefillScheduler(Scheduler):
    """Stock vLLM Scheduler with step-level signals; adaptive control follows in M2."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        raw_config = os.environ.get(SLOTUNE_SCHEDULER_CONFIG_ENV)
        self.adaptive_prefill_config = (
            AdaptivePrefillConfig.model_validate_json(raw_config)
            if raw_config
            else AdaptivePrefillConfig()
        )
        log_path = os.environ.get(SLOTUNE_SCHEDULER_LOG_ENV)
        self._decision_writer = (
            SchedulerDecisionWriter(log_path)
            if log_path and self.adaptive_prefill_config.decision_log_enabled
            else None
        )
        self._slotune_step_index = 0
        self._controller = AdaptivePrefillController(self.adaptive_prefill_config)
        self._active_prefill_cap = self.max_num_scheduled_tokens
        if self.adaptive_prefill_config.enabled and not hasattr(
            Scheduler, "_get_prefill_token_budget"
        ):
            raise RuntimeError("Adaptive Prefill requires the SLOTune vLLM v0.16.0 scheduler patch")

    def _get_prefill_token_budget(self) -> int:
        """Supply the current cap to the minimal patched vLLM scheduling loop."""
        return self._active_prefill_cap

    def _decode_token_reservation(self) -> int:
        """Reserve stock per-step work for every request already decoding."""
        reservation = 0
        for request in self.requests.values():
            if request.num_output_tokens == 0:
                continue
            demand = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            reservation += max(1, demand)
        return min(self.max_num_scheduled_tokens, reservation)

    def _controller_decision(
        self, signals: SchedulerSignals
    ) -> tuple[str, int, str, ControllerDecision | None]:
        if not self.adaptive_prefill_config.enabled:
            return "DISABLED", self.max_num_scheduled_tokens, "controller_disabled", None

        if self.adaptive_prefill_config.fixed_prefill_cap is not None:
            requested_cap = min(
                self.adaptive_prefill_config.fixed_prefill_cap,
                self.max_num_scheduled_tokens,
            )
            available_prefill_tokens = max(
                0,
                self.max_num_scheduled_tokens - self._decode_token_reservation(),
            )
            effective_cap = min(requested_cap, available_prefill_tokens)
            reason_code = "fixed_prefill_cap"
            if self.adaptive_prefill_config.fixed_prefill_cap > self.max_num_scheduled_tokens:
                reason_code = f"{reason_code};global_cap_clamped"
            if effective_cap < requested_cap:
                reason_code = f"{reason_code};decode_reservation_limited"
            return "FIXED", effective_cap, reason_code, None

        decision = self._controller.decide(signals)
        requested_cap = min(decision.prefill_cap, self.max_num_scheduled_tokens)
        decode_reservation = self._decode_token_reservation()
        available_prefill_tokens = max(0, self.max_num_scheduled_tokens - decode_reservation)
        effective_cap = min(requested_cap, available_prefill_tokens)
        reasons = [decision.reason_code]
        if decision.prefill_cap > self.max_num_scheduled_tokens:
            reasons.append("global_cap_clamped")
        if effective_cap < requested_cap:
            reasons.append("decode_reservation_limited")
        if (
            decision.max_wait_forced
            and effective_cap < self.adaptive_prefill_config.min_prefill_progress
        ):
            reasons.append("max_wait_progress_capacity_blocked")
        return decision.state.value, effective_cap, ";".join(reasons), decision

    def schedule(self) -> SchedulerOutput:
        """Delegate decisions to stock vLLM, then record signals and outcomes."""
        started_ns = time.perf_counter_ns()
        signals = collect_scheduler_signals(
            self.requests,
            now=time.time(),
            kv_cache_usage=self.kv_cache_manager.usage,
            running_requests=len(self.running),
            waiting_requests=len(self.waiting),
        )
        controller_state, prefill_cap, reason_code, controller_decision = self._controller_decision(
            signals
        )
        self._active_prefill_cap = prefill_cap
        scheduler_output = super().schedule()
        scheduled_decode_tokens, scheduled_prefill_tokens = split_scheduled_tokens(
            scheduler_output.num_scheduled_tokens,
            signals.prefill_request_ids,
        )
        if self.adaptive_prefill_config.enabled:
            assert scheduled_prefill_tokens <= prefill_cap, (
                "Patched vLLM Scheduler exceeded the selected Prefill cap: "
                f"{scheduled_prefill_tokens} > {prefill_cap}"
            )
        if (
            controller_decision is not None
            and controller_decision.max_wait_forced
            and scheduled_prefill_tokens < self.adaptive_prefill_config.min_prefill_progress
        ):
            progress_outcome = (
                "max_wait_progress_not_met"
                if has_unfinished_prefill(self.requests, signals.prefill_request_ids)
                else "max_wait_remaining_completed"
            )
            reason_code = f"{reason_code};{progress_outcome}"
        elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.0
        if self._decision_writer is not None:
            self._decision_writer.write(
                SchedulerStepRecord(
                    timestamp=signals.timestamp,
                    step_index=self._slotune_step_index,
                    controller_state=controller_state,
                    decode_backlog=signals.decode_backlog,
                    oldest_prefill_wait_ms=signals.oldest_prefill_wait_ms,
                    kv_cache_usage=signals.kv_cache_usage,
                    prefill_cap=prefill_cap,
                    scheduled_decode_tokens=scheduled_decode_tokens,
                    scheduled_prefill_tokens=scheduled_prefill_tokens,
                    total_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
                    running_requests=signals.running_requests,
                    waiting_requests=signals.waiting_requests,
                    preemption_delta=len(scheduler_output.preempted_req_ids or ()),
                    scheduler_cpu_time_us=elapsed_us,
                    reason_code=reason_code,
                )
            )
        self._slotune_step_index += 1
        return scheduler_output

    def shutdown(self) -> None:
        """Close the decision log even when vLLM connector shutdown fails."""
        try:
            super().shutdown()
        finally:
            if self._decision_writer is not None:
                self._decision_writer.close()
