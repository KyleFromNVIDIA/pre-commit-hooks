# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import re
from dataclasses import dataclass, field
from functools import cache
from typing import Optional, TYPE_CHECKING

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

from rapids_pre_commit_hooks.utils.dependencies_yaml import (
    Handler,
)
from rapids_metadata.remote import fetch_latest

if TYPE_CHECKING:
    import argparse
    from collections.abc import Generator

    import yaml

    from rapids_pre_commit_hooks.lint import Linter
    from rapids_metadata.metadata import RAPIDSMetadata, RAPIDSVersion


@cache
def all_metadata() -> "RAPIDSMetadata":
    return fetch_latest()


def get_rapids_version(args: "argparse.Namespace") -> "RAPIDSVersion":
    md = all_metadata()
    return (
        md.versions[args.rapids_version]
        if args.rapids_version
        else md.get_current_version(os.getcwd(), args.rapids_version_file)
    )


class CUDASuffixedHandler(Handler):
    @dataclass
    class CommonContext:
        common_key: "yaml.Node"

    @dataclass
    class CommonItemContext:
        has_python_output_type: bool = False
        suspicious_suffixed_packages: "list[tuple[str, Optional[str], yaml.Node]]" = field(  # noqa: E501
            default_factory=list
        )
        suspicious_unsuffixed_packages: "list[tuple[str, Optional[str], yaml.Node]]" = field(  # noqa: E501
            default_factory=list
        )

    @dataclass
    class SpecificItemContext:
        has_python_output_type: bool = False
        matrices_item_contexts: "list[CUDASuffixedHandler.MatricesItemContext]" = field(  # noqa: E501
            default_factory=list
        )

    @dataclass
    class MatricesItemContext:
        matrix_node: "Optional[yaml.Node]" = None
        cuda_suffixed_node: "Optional[yaml.Node]" = None
        cuda_suffixed: "Optional[bool]" = None
        cuda_node: "Optional[yaml.Node]" = None
        cuda_major: "Optional[int]" = None
        suspicious_suffixed_packages: "list[tuple[str, Optional[str], yaml.Node]]" = field(  # noqa: E501
            default_factory=list
        )
        suspicious_unsuffixed_packages: "list[tuple[str, Optional[str], yaml.Node]]" = field(  # noqa: E501
            default_factory=list
        )

    def __init__(self, linter: "Linter", args: "argparse.Namespace") -> None:
        self.linter = linter
        self.args = args

    def handle_output_type(
        self,
        output_types_context: (
            "CUDASuffixedHandler.CommonItemContext | "
            "CUDASuffixedHandler.SpecificItemContext"
        ),
        item: "yaml.Node",
    ) -> None:
        if item.value in {"requirements", "pyproject"}:
            output_types_context.has_python_output_type = True

    @contextlib.contextmanager
    def handle_common(
        self,
        dependency_set_context: None,  # noqa: ARG002
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[CUDASuffixedHandler.CommonContext]":
        context = CUDASuffixedHandler.CommonContext(key)
        yield context

    @contextlib.contextmanager
    def handle_common_item(
        self,
        common_context: "CUDASuffixedHandler.CommonContext",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[CUDASuffixedHandler.CommonItemContext]":
        context = CUDASuffixedHandler.CommonItemContext()
        yield context

        if context.has_python_output_type:
            for name, anchor, node in context.suspicious_suffixed_packages:
                w = self.linter.add_warning(
                    (node.start_mark.index, node.end_mark.index),
                    f'package "{name}" in common dependency set',
                )
                w.add_note(
                    (
                        common_context.common_key.start_mark.index,
                        common_context.common_key.end_mark.index,
                    ),
                    "place in a specific dependency set with "
                    'cuda_suffixed: "true" instead',
                )
            for name, anchor, node in context.suspicious_unsuffixed_packages:
                w = self.linter.add_warning(
                    (node.start_mark.index, node.end_mark.index),
                    f'package "{name}" in common dependency set',
                )
                w.add_note(
                    (
                        common_context.common_key.start_mark.index,
                        common_context.common_key.end_mark.index,
                    ),
                    "place in a specific dependency set with "
                    'cuda_suffixed: "false" instead',
                )

    @contextlib.contextmanager
    def handle_specific_item(
        self,
        specific_context: None,  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[CUDASuffixedHandler.SpecificItemContext]":
        context = CUDASuffixedHandler.SpecificItemContext()
        yield context

        if context.has_python_output_type:
            for matrices_item_context in context.matrices_item_contexts:
                if matrices_item_context.cuda_suffixed is None:
                    for (
                        name,
                        anchor,
                        node,
                    ) in matrices_item_context.suspicious_suffixed_packages:
                        w = self.linter.add_warning(
                            (node.start_mark.index, node.end_mark.index),
                            f'package "{name}" in specific dependency set '
                            "with no cuda_suffixed field",
                        )
                        if matrices_item_context.matrix_node:
                            w.add_note(
                                (
                                    matrices_item_context.matrix_node.start_mark.index,
                                    matrices_item_context.matrix_node.end_mark.index,
                                ),
                                "place in a specific dependency set with "
                                'cuda_suffixed: "true" instead',
                            )
                    for (
                        name,
                        anchor,
                        node,
                    ) in matrices_item_context.suspicious_unsuffixed_packages:
                        w = self.linter.add_warning(
                            (node.start_mark.index, node.end_mark.index),
                            f'package "{name}" in common dependency set',
                        )
                        if matrices_item_context.matrix_node:
                            w.add_note(
                                (
                                    matrices_item_context.matrix_node.start_mark.index,
                                    matrices_item_context.matrix_node.end_mark.index,
                                ),
                                "place in a specific dependency set with "
                                'cuda_suffixed: "false" instead',
                            )
                elif matrices_item_context.cuda_suffixed:
                    for (
                        name,
                        anchor,
                        node,
                    ) in matrices_item_context.suspicious_unsuffixed_packages:
                        w = self.linter.add_warning(
                            (node.start_mark.index, node.end_mark.index),
                            f'package "{name}" in specific dependency set '
                            'with cuda_suffixed: "true"',
                        )
                        if matrices_item_context.cuda_major:
                            anchor_text = f"&{anchor} " if anchor else ""
                            w.add_replacement(
                                (node.start_mark.index, node.end_mark.index),
                                f"{anchor_text}{name}-cu{matrices_item_context.cuda_major}",
                            )
                        elif matrices_item_context.matrix_node:
                            w.add_note(
                                (
                                    matrices_item_context.matrix_node.start_mark.index,
                                    matrices_item_context.matrix_node.end_mark.index,
                                ),
                                "add a cuda matrix field and add matching "
                                "-cu* suffix to package name",
                            )
                else:
                    for (
                        name,
                        anchor,
                        node,
                    ) in matrices_item_context.suspicious_suffixed_packages:
                        w = self.linter.add_warning(
                            (node.start_mark.index, node.end_mark.index),
                            f'package "{name}" in specific dependency set '
                            'with cuda_suffixed: "false"',
                        )
                        anchor_text = f"&{anchor} " if anchor else ""
                        w.add_replacement(
                            (node.start_mark.index, node.end_mark.index),
                            f"{anchor_text}{name}",
                        )

    @contextlib.contextmanager
    def handle_matrices_item(
        self,
        matrices_context: "CUDASuffixedHandler.SpecificItemContext",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[CUDASuffixedHandler.MatricesItemContext]":
        context = CUDASuffixedHandler.MatricesItemContext()
        yield context

        matrices_context.matrices_item_contexts.append(context)

    @contextlib.contextmanager
    def handle_matrix(
        self,
        matrices_item_context: "CUDASuffixedHandler.MatricesItemContext",
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[CUDASuffixedHandler.MatricesItemContext]":
        matrices_item_context.matrix_node = key
        yield matrices_item_context

    def handle_matrix_item(
        self,
        matrix_context: "CUDASuffixedHandler.MatricesItemContext",
        key: "yaml.Node",
        value: "yaml.Node",
    ) -> None:
        if key.value == "cuda_suffixed":
            matrix_context.cuda_suffixed_node = value
            if value.value == "true":
                matrix_context.cuda_suffixed = True
            elif value.value == "false":
                matrix_context.cuda_suffixed = False
        elif key.value == "cuda":
            matrix_context.cuda_node = value
            matrix_context.cuda_major = Version(value.value).major

    def handle_package(
        self,
        packages_context: (
            "CUDASuffixedHandler.CommonItemContext | "
            "CUDASuffixedHandler.MatricesItemContext"
        ),
        anchor: "Optional[str]",
        item: "yaml.Node",
    ) -> None:
        try:
            req = Requirement(item.value)
        except InvalidRequirement:
            return

        if req.name in get_rapids_version(self.args).cuda_suffixed_packages:
            packages_context.suspicious_unsuffixed_packages.append(
                (req.name, anchor, item)
            )
        elif (
            match := re.search(r"^(?P<package>.*)-cu[0-9]+$", req.name)
        ) and match.group("package") in get_rapids_version(
            self.args
        ).cuda_suffixed_packages:
            packages_context.suspicious_suffixed_packages.append(
                (match.group("package"), anchor, item)
            )
