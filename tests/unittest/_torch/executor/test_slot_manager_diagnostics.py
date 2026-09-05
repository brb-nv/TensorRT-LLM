# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SlotManager occupancy and exhaustion reporting.

Slot-pool exhaustion is raised on the executor's event-loop thread, so the
exception text is the only artifact that survives: the rank stops joining
collectives and the job is torn down by the hang detector minutes later, on a
different rank. Occupancy alone cannot say whether the pool was merely
oversubscribed for an iteration or is leaking seats, so the report carries
holder ages as well as counts.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import (NoFreeSlotsError,
                                                             SlotManager)


def test_occupancy_str_reports_used_total_and_oldest_age():
    manager = SlotManager(3, name="seq_slots")

    assert manager.occupancy_str() == "0/3"

    manager.add_slot(101)
    occupancy = manager.occupancy_str()

    assert occupancy.startswith("1/3(oldest=")
    assert occupancy.endswith("s)")


def test_exhaustion_report_names_the_pool_and_every_holder():
    manager = SlotManager(2, name="seq_slots")
    manager.add_slot(101)
    manager.add_slot(102)

    with pytest.raises(NoFreeSlotsError) as excinfo:
        manager.add_slot(103)

    report = str(excinfo.value)
    assert "'seq_slots'" in report
    assert "2/2 held" in report
    assert "req=101@slot" in report
    assert "req=102@slot" in report


def test_freeing_a_slot_drops_its_holder_from_the_report():
    manager = SlotManager(2, name="seq_slots")
    manager.add_slot(101)
    manager.add_slot(102)
    manager.remove_slot(101)

    assert manager.occupancy_str().startswith("1/2(oldest=")
    manager.add_slot(103)
    with pytest.raises(NoFreeSlotsError, match="req=102@slot"):
        manager.add_slot(104)


def test_shutdown_drains_holder_ages():
    """A stale acquired_at entry would misreport the oldest age after reuse."""
    manager = SlotManager(2, name="seq_slots")
    manager.add_slot(101)
    manager.add_slot(102)

    manager.shutdown()

    assert manager.occupancy_str() == "0/2"
    assert not manager.acquired_at
