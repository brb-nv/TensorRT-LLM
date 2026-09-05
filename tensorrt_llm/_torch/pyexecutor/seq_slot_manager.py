from tensorrt_llm.logger import logger

from .llm_request import LlmRequest
from .resource_manager import (BaseResourceManager, NoFreeSlotsError,
                               SlotManager)
from .scheduler import ScheduledRequests


class SeqSlotManager(BaseResourceManager):

    def __init__(self, max_num_sequences: int):
        self.slot_manager = SlotManager(max_num_sequences, name="seq_slots")

    def get_max_resource_count(self) -> int:
        return self.slot_manager.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 1

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        for llm_req in scheduled_batch.all_requests():
            if llm_req.is_disagg_generation_init_state:
                # Per request per iteration, so debug rather than info.
                logger.debug(
                    f"Skip assigning sequence slot for DISAGG_GENERATION_INIT "
                    f"request {llm_req.request_id}.")
                continue
            if llm_req.seq_slot is None or llm_req.is_disagg_generation_transmission_complete:
                try:
                    llm_req.seq_slot = self.slot_manager.add_slot(
                        llm_req.request_id)
                except NoFreeSlotsError as e:
                    raise NoFreeSlotsError(
                        f"{e} while admitting {self._batch_summary(scheduled_batch)}"
                    ) from e
                llm_req.py_seq_slot = llm_req.seq_slot
                if llm_req.return_perf_metrics:
                    llm_req.set_first_scheduled_time()

    @staticmethod
    def _batch_summary(scheduled_batch: ScheduledRequests) -> str:
        """Describe the batch that could not be admitted.

        Distinguishes an oversubscribed pool (the batch itself exceeds the pool)
        from stale holders (a small batch cannot be admitted anyway).
        """
        requests = list(scheduled_batch.all_requests())
        transmission_complete = sum(
            1 for r in requests if r.is_disagg_generation_transmission_complete)
        already_held = sum(1 for r in requests if r.seq_slot is not None)
        return (f"batch of {len(requests)} "
                f"(disagg_transmission_complete={transmission_complete}, "
                f"already_holding_slot={already_held}, "
                f"req_ids={[r.request_id for r in requests]})")

    def free_resources(self, request: LlmRequest) -> None:
        self.slot_manager.remove_slot(request.request_id)
