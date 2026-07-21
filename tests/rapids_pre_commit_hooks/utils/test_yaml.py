# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from rapids_pre_commit_hooks.utils.yaml import (
    AnchorPreservingLoader,
    check_and_mark_anchor,
)


def test_anchor_preserving_loader():
    loader = AnchorPreservingLoader("- &a A\n- *a")
    try:
        root = loader.get_single_node()
    finally:
        loader.dispose()
    assert loader.document_anchors == [{"a": root.value[0]}]


@pytest.mark.parametrize(
    [
        "used_anchors_before",
        "node_index",
        "descend",
        "anchor",
        "used_anchors_after",
    ],
    [
        (
            set(),
            0,
            True,
            "anchor1",
            {"anchor1"},
        ),
        (
            {"anchor1"},
            1,
            True,
            "anchor2",
            {"anchor1", "anchor2"},
        ),
        (
            set(),
            2,
            True,
            None,
            set(),
        ),
        (
            {"anchor1", "anchor2"},
            0,
            False,
            "anchor1",
            {"anchor1", "anchor2"},
        ),
        (
            {"anchor1", "anchor2"},
            1,
            False,
            "anchor2",
            {"anchor1", "anchor2"},
        ),
    ],
)
def test_check_and_mark_anchor(
    used_anchors_before,
    node_index,
    descend,
    anchor,
    used_anchors_after,
):
    NODES = [Mock() for _ in range(3)]
    ANCHORS = {
        "anchor1": NODES[0],
        "anchor2": NODES[1],
    }
    used_anchors = set(used_anchors_before)
    actual_descend, actual_anchor = check_and_mark_anchor(
        ANCHORS, used_anchors, NODES[node_index]
    )
    assert actual_descend == descend
    assert actual_anchor == anchor
    assert used_anchors == used_anchors_after
