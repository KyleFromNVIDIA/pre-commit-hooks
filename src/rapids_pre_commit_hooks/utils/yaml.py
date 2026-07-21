# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import yaml


class AnchorPreservingLoader(yaml.SafeLoader):
    """A SafeLoader that preserves the anchors for later reference. The anchors
    can be found in the document_anchors member, which is a list of
    dictionaries, one dictionary for each parsed document.
    """

    def __init__(self, stream) -> None:
        super().__init__(stream)
        self.document_anchors: list[dict[str, yaml.Node]] = []

    def compose_document(self) -> "yaml.Node":
        # Drop the DOCUMENT-START event.
        self.get_event()

        # Compose the root node.
        node = self.compose_node(None, None)  # type: ignore[arg-type]

        # Drop the DOCUMENT-END event.
        self.get_event()

        self.document_anchors.append(self.anchors)
        self.anchors = {}
        assert node is not None
        return node


def node_has_type(node: "yaml.Node", tag_type: str) -> bool:
    return node.tag == f"tag:yaml.org,2002:{tag_type}"


def check_and_mark_anchor(
    anchors: dict[str, "yaml.Node"], used_anchors: set[str], node: "yaml.Node"
) -> tuple[bool, str | None]:
    for key, value in anchors.items():
        if value == node:
            anchor = key
            break
    else:
        anchor = None
    if anchor in used_anchors:
        return False, anchor
    if anchor is not None:
        used_anchors.add(anchor)
    return True, anchor
