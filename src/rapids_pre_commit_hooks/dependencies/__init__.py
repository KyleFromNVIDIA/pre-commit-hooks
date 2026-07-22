# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from .use_cuda_wheels import UseCUDAWheelsHandler
from ..lint import Linter, LintMain
from ..utils.dependencies_yaml import (
    ChainedHandler,
    traverse_dependencies_yaml,
)


def check_dependencies(linter: "Linter", args: "argparse.Namespace") -> None:
    handler = ChainedHandler()
    handler.add_handler(UseCUDAWheelsHandler(linter, args))
    traverse_dependencies_yaml(handler, linter.content)


def main() -> None:
    m = LintMain("verify-dependencies")
    m.argparser.description = (
        "Verify that dependencies.yaml follows the correct conventions."
    )
    with m.execute() as ctx:
        ctx.add_check(check_dependencies)


if __name__ == "__main__":
    main()
