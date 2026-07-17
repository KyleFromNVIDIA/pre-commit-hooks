# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Any, Optional

import yaml


class Handler:
    def handle_root(
        self,
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext()

    def handle_dependencies(
        self,
        root_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(root_context)

    def handle_dependency_set(
        self,
        dependencies_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(dependencies_context)

    def handle_common(
        self,
        dependency_set_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(dependency_set_context)

    def handle_common_item(
        self,
        common_context: "Any",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(common_context)

    def handle_specific(
        self,
        dependency_set_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(dependency_set_context)

    def handle_specific_item(
        self,
        specific_context: "Any",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(specific_context)

    def handle_matrices(
        self,
        specific_item_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(specific_item_context)

    def handle_matrices_item(
        self,
        matrices_context: "Any",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(matrices_context)

    def handle_packages(
        self,
        common_or_matrices_item_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(common_or_matrices_item_context)

    def handle_package(
        self,
        packages_context: "Any",  # noqa: ARG002
        anchor: "Optional[str]",  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> None:
        pass


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


def traverse_package(
    handler: Handler,
    packages_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "str"):
        descend, anchor = check_and_mark_anchor(anchors, used_anchors, node)
        if descend:
            handler.handle_package(packages_context, anchor, node)


def traverse_packages(
    handler: Handler,
    common_or_matrices_item_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        descend, _ = check_and_mark_anchor(anchors, used_anchors, node)
        if descend:
            with handler.handle_packages(
                common_or_matrices_item_context, key_node, node
            ) as packages_context:
                for package in node.value:
                    traverse_package(
                        handler,
                        packages_context,
                        anchors,
                        used_anchors,
                        package,
                    )


def traverse_common_item(
    handler: Handler,
    common_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_common_item(
            common_context, node
        ) as common_item_context:
            for (
                common_item_key,
                common_item_value,
            ) in node.value:
                if (
                    node_has_type(common_item_key, "str")
                    and common_item_key.value == "packages"
                ):
                    traverse_packages(
                        handler,
                        common_item_context,
                        anchors,
                        used_anchors,
                        common_item_key,
                        common_item_value,
                    )


def traverse_common(
    handler: Handler,
    dependency_set_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        with handler.handle_common(
            dependency_set_context, key_node, node
        ) as common_context:
            for common_item in node.value:
                traverse_common_item(
                    handler, common_context, anchors, used_anchors, common_item
                )


def traverse_matrices_item(
    handler: Handler,
    matrices_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_matrices_item(
            matrices_context, node
        ) as matrices_item_context:
            for matrix_key, matrix_value in node.value:
                if (
                    node_has_type(matrix_key, "str")
                    and matrix_key.value == "packages"
                ):
                    traverse_packages(
                        handler,
                        matrices_item_context,
                        anchors,
                        used_anchors,
                        matrix_key,
                        matrix_value,
                    )


def traverse_matrices(
    handler: Handler,
    specific_item_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        with handler.handle_matrices(
            specific_item_context, key_node, node
        ) as matrices_context:
            for item in node.value:
                traverse_matrices_item(
                    handler, matrices_context, anchors, used_anchors, item
                )


def traverse_specific_item(
    handler: Handler,
    specific_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_specific_item(
            specific_context, node
        ) as specific_item_context:
            for (
                specific_item_key,
                specific_item_value,
            ) in node.value:
                if (
                    node_has_type(specific_item_key, "str")
                    and specific_item_key.value == "matrices"
                ):
                    traverse_matrices(
                        handler,
                        specific_item_context,
                        anchors,
                        used_anchors,
                        specific_item_key,
                        specific_item_value,
                    )


def traverse_specific(
    handler: Handler,
    dependency_set_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        with handler.handle_specific(
            dependency_set_context, key_node, node
        ) as specific_context:
            for specific_item in node.value:
                traverse_specific_item(
                    handler,
                    specific_context,
                    anchors,
                    used_anchors,
                    specific_item,
                )


def traverse_dependency_set(
    handler: Handler,
    dependencies_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_dependency_set(
            dependencies_context, key_node, node
        ) as dependency_set_context:
            for (
                dependency_key,
                dependency_value,
            ) in node.value:
                if node_has_type(dependency_key, "str"):
                    if dependency_key.value == "common":
                        traverse_common(
                            handler,
                            dependency_set_context,
                            anchors,
                            used_anchors,
                            dependency_key,
                            dependency_value,
                        )
                    elif dependency_key.value == "specific":
                        traverse_specific(
                            handler,
                            dependency_set_context,
                            anchors,
                            used_anchors,
                            dependency_key,
                            dependency_value,
                        )


def traverse_dependencies(
    handler: Handler,
    root_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_dependencies(
            root_context, key_node, node
        ) as dependencies_context:
            for dependencies_key, dependencies_value in node.value:
                traverse_dependency_set(
                    handler,
                    dependencies_context,
                    anchors,
                    used_anchors,
                    dependencies_key,
                    dependencies_value,
                )


def traverse_root(
    handler: Handler,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_root(node) as root_context:
            for root_key, root_value in node.value:
                if (
                    node_has_type(root_key, "str")
                    and root_key.value == "dependencies"
                ):
                    traverse_dependencies(
                        handler,
                        root_context,
                        anchors,
                        used_anchors,
                        root_key,
                        root_value,
                    )


def traverse_dependencies_yaml(handler: Handler, content: str) -> None:
    loader = AnchorPreservingLoader(content)
    try:
        root = loader.get_single_node()
        assert root is not None
    finally:
        loader.dispose()
    traverse_root(handler, loader.document_anchors[0], set(), root)
