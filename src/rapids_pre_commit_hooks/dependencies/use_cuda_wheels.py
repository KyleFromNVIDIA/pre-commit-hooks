# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from packaging.requirements import InvalidRequirement, Requirement
from rapids_pre_commit_hooks.utils.dependencies_yaml import (
    Handler,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import yaml

    from rapids_pre_commit_hooks.lint import Linter


class UseCUDAWheelsHandler(Handler):
    @dataclass
    class Context:
        has_use_cuda_wheels: bool = False
        use_cuda_wheels_node: "Optional[yaml.Node]" = None
        suspicious_packages: "list[tuple[yaml.Node, str]]" = field(
            default_factory=list
        )

    def __init__(self, linter: "Linter", args: argparse.Namespace):
        self.linter = linter
        self.args = args

    @contextlib.contextmanager
    def handle_common(
        self,
        dependency_set_context: "Any",  # noqa: ARG002
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.Context]":
        context = UseCUDAWheelsHandler.Context()
        yield context

        for node, name in context.suspicious_packages:
            w = self.linter.add_warning(
                (node.start_mark.index, node.end_mark.index),
                f'package "{name}" in common dependency set',
            )
            w.add_note(
                (key.start_mark.index, key.end_mark.index),
                "place in a specific dependency set with "
                'use_cuda_wheels: "true" instead',
            )

    @contextlib.contextmanager
    def handle_matrices_item(
        self,
        matrices_context: "Any",  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.Context]":
        context = UseCUDAWheelsHandler.Context()
        yield context

        if not context.has_use_cuda_wheels:
            for node, name in context.suspicious_packages:
                w = self.linter.add_warning(
                    (node.start_mark.index, node.end_mark.index),
                    f'package "{name}" in specific dependency set without '
                    'use_cuda_wheels: "true"',
                )
                if context.use_cuda_wheels_node:
                    w.add_note(
                        (
                            context.use_cuda_wheels_node.start_mark.index,
                            context.use_cuda_wheels_node.end_mark.index,
                        ),
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    )

    @contextlib.contextmanager
    def handle_matrix(
        self,
        matrices_item_context: "UseCUDAWheelsHandler.Context",
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.Context]":
        matrices_item_context.use_cuda_wheels_node = key
        yield matrices_item_context

    def handle_matrix_item(
        self,
        matrix_context: "UseCUDAWheelsHandler.Context",
        key: "yaml.Node",
        value: "yaml.Node",
    ) -> None:
        if key.value == "use_cuda_wheels":
            matrix_context.use_cuda_wheels_node = value
            if value.value == "true":
                matrix_context.has_use_cuda_wheels = True

    @contextlib.contextmanager
    def handle_packages(
        self,
        common_or_matrices_item_context: "UseCUDAWheelsHandler.Context",
        key: "yaml.Node",
        value: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[UseCUDAWheelsHandler.Context]":
        if common_or_matrices_item_context.use_cuda_wheels_node is None:
            common_or_matrices_item_context.use_cuda_wheels_node = key
        yield common_or_matrices_item_context

    def handle_package(
        self,
        packages_context: "UseCUDAWheelsHandler.Context",
        anchor: "Optional[str]",  # noqa: ARG002
        item: "yaml.Node",
    ) -> None:
        try:
            req = Requirement(item.value)
        except InvalidRequirement:
            return
        if req.name == "cuda-toolkit":
            packages_context.suspicious_packages.append((item, req.name))
        elif (
            re.search(r"^cupy-cuda[0-9]+x$", req.name) and "ctk" in req.extras
        ):
            packages_context.suspicious_packages.append(
                (item, f"{req.name}[ctk]")
            )
