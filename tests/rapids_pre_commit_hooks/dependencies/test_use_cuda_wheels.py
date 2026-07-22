# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement

from rapids_pre_commit_hooks import lint, dependencies
from rapids_pre_commit_hooks.dependencies.use_cuda_wheels import (
    UseCUDAWheelsHandler,
    is_cupy_ctk_package,
    is_nvidia_library_package,
)
from rapids_pre_commit_hooks.utils import dependencies_yaml
from rapids_pre_commit_hooks_test_utils import (
    find_yaml_node_for_span,
    parse_named_spans,
)


@pytest.mark.parametrize(
    ["name", "expected_result"],
    [
        pytest.param("cuda-toolkit", True, id="cuda-toolkit"),
        pytest.param("cuda-toolkit[cufile]", True, id="cuda-toolkit-extra"),
        pytest.param("cuda-toolkit-cu12", False, id="cuda-toolkit-cu12"),
        pytest.param("nvidia-curand", True, id="nvidia-curand"),
        pytest.param("nvidia-curand-cu12", True, id="nvidia-curand-cu12"),
        pytest.param("nvidia-curand-cu13", True, id="nvidia-curand-cu13"),
        pytest.param(
            "nvidia-curand-cu13a", False, id="nvidia-curand-cu13-suffix"
        ),
        pytest.param(
            "anvidia-curand-cu13", False, id="nvidia-curand-cu13-prefix"
        ),
        pytest.param("other-package", False, id="other-package"),
        pytest.param("other-package-cu13", False, id="other-package-cu13"),
    ],
)
def test_is_nvidia_library_package(name, expected_result):
    assert is_nvidia_library_package(Requirement(name)) == expected_result


@pytest.mark.parametrize(
    ["name", "expected_result"],
    [
        pytest.param("cupy-cuda12x[ctk]", True, id="cupy-cuda12x-ctk"),
        pytest.param("cupy-cuda13x[ctk]", True, id="cupy-cuda13x-ctk"),
        pytest.param(
            "cupy-cuda12x[ctk,other]", True, id="cupy-cuda12x-ctk-and-other"
        ),
        pytest.param("cupy-cuda12x[other]", False, id="cupy-cuda12x-other"),
        pytest.param("cupy-cuda12x", False, id="cupy-cuda12x-no-extras"),
        pytest.param("other-package", False, id="other-package"),
        pytest.param("other-package[ctk]", False, id="other-package-ctk"),
    ],
)
def test_is_cupy_ctk_package(name, expected_result):
    assert is_cupy_ctk_package(Requirement(name)) == expected_result


class TestUseCUDAWheelsHandler:
    @pytest.mark.parametrize(
        ["content", "suspicious_package_names", "expected_warnings"],
        [
            pytest.param(
                """\
                + common:
                +   packages:
                +     - package1
                +     - package2
                +     - package3
                """,
                [],
                [],
                id="no-suspicious-packages",
            ),
            pytest.param(
                """\
                + common:
                : ~~~~~~warnings.0.notes.0
                +   packages:
                +     - package1
                +     - package2
                :       ~~~~~~~~packages.0
                :       ~~~~~~~~warnings.0.warning
                +     - package3
                """,
                ["package2"],
                [
                    {
                        "warning": 'package "package2" in common dependency '
                        "set",
                        "notes": [
                            "place in a specific dependency set with "
                            'use_cuda_wheels: "true" instead',
                        ],
                    },
                ],
                id="suspicious-packages",
            ),
        ],
    )
    def test_handle_common(
        self, content, suspicious_package_names, expected_warnings
    ):
        content, spans = parse_named_spans(content, dict)

        args = Mock()
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-use-cuda-wheels"
        )
        loader = dependencies_yaml.AnchorPreservingLoader(content)
        try:
            composed = loader.get_single_node()
        finally:
            loader.dispose()
        common_key, common = composed.value[0]

        handler = UseCUDAWheelsHandler(linter, args)
        with handler.handle_common(
            Mock(), common_key, common
        ) as common_context:
            common_context.suspicious_packages.extend(
                zip(
                    (
                        find_yaml_node_for_span(composed, span)
                        for span in spans.get("packages", [])
                    ),
                    suspicious_package_names,
                    strict=True,
                )
            )

        assert linter.warnings == [
            lint.LintWarning(
                warning_span["warning"],
                warning["warning"],
                notes=[
                    lint.Note(note_span, note)
                    for note_span, note in zip(
                        warning_span.get("notes", []),
                        warning["notes"],
                        strict=True,
                    )
                ],
            )
            for warning_span, warning in zip(
                spans.get("warnings", []), expected_warnings, strict=True
            )
        ]

    @pytest.mark.parametrize(
        [
            "content",
            "has_use_cuda_wheels",
            "suspicious_package_names",
            "expected_warnings",
        ],
        [
            pytest.param(
                """\
                + matrix:
                +   use_cuda_wheels: "true"
                :                    ~~~~~~use_cuda_wheels
                + packages:
                +   - package1
                +   - package2
                :     ~~~~~~~~packages.0
                +   - package3
                """,
                True,
                ["package2"],
                [],
                id="use-cuda-wheels-true",
            ),
            pytest.param(
                """\
                + matrix:
                +   use_cuda_wheels: "false"
                :                    ~~~~~~~use_cuda_wheels
                :                    ~~~~~~~warnings.0.notes.0
                + packages:
                +   - package1
                +   - package2
                :     ~~~~~~~~packages.0
                :     ~~~~~~~~warnings.0.warning
                +   - package3
                """,
                False,
                ["package2"],
                [
                    {
                        "warning": 'package "package2" in specific dependency '
                        'set without use_cuda_wheels: "true"',
                        "notes": [
                            "place in a specific dependency set with "
                            'use_cuda_wheels: "true" instead',
                        ],
                    },
                ],
                id="use-cuda-wheels-false",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~use_cuda_wheels
                : ~~~~~~warnings.0.notes.0
                +   other_key: "other_value"
                + packages:
                +   - package1
                +   - package2
                :     ~~~~~~~~packages.0
                :     ~~~~~~~~warnings.0.warning
                +   - package3
                """,
                False,
                ["package2"],
                [
                    {
                        "warning": 'package "package2" in specific dependency '
                        'set without use_cuda_wheels: "true"',
                        "notes": [
                            "place in a specific dependency set with "
                            'use_cuda_wheels: "true" instead',
                        ],
                    },
                ],
                id="no-use-cuda-wheels",
            ),
            pytest.param(
                """\
                + packages:
                : ~~~~~~~~use_cuda_wheels
                : ~~~~~~~~warnings.0.notes.0
                +   - package1
                +   - package2
                :     ~~~~~~~~packages.0
                :     ~~~~~~~~warnings.0.warning
                +   - package3
                """,
                False,
                ["package2"],
                [
                    {
                        "warning": 'package "package2" in specific dependency '
                        'set without use_cuda_wheels: "true"',
                        "notes": [
                            "place in a specific dependency set with "
                            'use_cuda_wheels: "true" instead',
                        ],
                    },
                ],
                id="no-matrix",
            ),
            pytest.param(
                """\
                + packages:
                : ~~~~~~~~use_cuda_wheels
                +   - package1
                +   - package2
                +   - package3
                """,
                False,
                [],
                [],
                id="no-suspicious-packages",
            ),
        ],
    )
    def test_handle_matrices_item(
        self,
        content,
        has_use_cuda_wheels,
        suspicious_package_names,
        expected_warnings,
    ):
        content, spans = parse_named_spans(content, dict)

        args = Mock()
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-use-cuda-wheels"
        )
        loader = dependencies_yaml.AnchorPreservingLoader(content)
        try:
            composed = loader.get_single_node()
        finally:
            loader.dispose()

        handler = UseCUDAWheelsHandler(linter, args)
        with handler.handle_matrices_item(
            Mock(), composed
        ) as matrices_item_context:
            matrices_item_context.has_use_cuda_wheels = has_use_cuda_wheels
            matrices_item_context.use_cuda_wheels_node = (
                find_yaml_node_for_span(composed, use_cuda_wheels_span)
                if (use_cuda_wheels_span := spans.get("use_cuda_wheels"))
                else None
            )
            matrices_item_context.suspicious_packages.extend(
                zip(
                    (
                        find_yaml_node_for_span(composed, span)
                        for span in spans.get("packages", [])
                    ),
                    suspicious_package_names,
                    strict=True,
                )
            )

        assert linter.warnings == [
            lint.LintWarning(
                warning_span["warning"],
                warning["warning"],
                notes=[
                    lint.Note(note_span, note)
                    for note_span, note in zip(
                        warning_span.get("notes", []),
                        warning["notes"],
                        strict=True,
                    )
                ],
            )
            for warning_span, warning in zip(
                spans.get("warnings", []), expected_warnings, strict=True
            )
        ]

    def test_handle_matrix(self):
        content, spans = parse_named_spans(
            """\
            + packages: []
            : ~~~~~~~~original_node
            + matrix:
            : ~~~~~~node
            : ~~~~~~matrix_key
            +     key_1: value_1
            :     >matrix
            +     key_2: value_2
            :                    !matrix
            """
        )

        args = Mock()
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-use-cuda-wheels"
        )
        loader = dependencies_yaml.AnchorPreservingLoader(content)
        try:
            composed = loader.get_single_node()
        finally:
            loader.dispose()
        matrix_key = find_yaml_node_for_span(composed, spans["matrix_key"])
        matrix = find_yaml_node_for_span(composed, spans["matrix"])

        handler = UseCUDAWheelsHandler(linter, args)
        original_node = find_yaml_node_for_span(
            composed, spans["original_node"]
        )
        node = find_yaml_node_for_span(composed, spans["node"])
        with handler.handle_matrix(
            Mock(use_cuda_wheels_node=original_node), matrix_key, matrix
        ) as matrix_context:
            assert matrix_context.use_cuda_wheels_node == node

    @pytest.mark.parametrize(
        ["content", "expected_has_use_cuda_wheels"],
        [
            pytest.param(
                """\
                + use_cuda_wheels: "false"
                :                  ~~~~~~~node
                """,
                False,
                id="use-cuda-wheels-false",
            ),
            pytest.param(
                """\
                + use_cuda_wheels: "true"
                :                  ~~~~~~node
                """,
                True,
                id="use-cuda-wheels-true",
            ),
            pytest.param(
                """\
                + other_key: "other_value"
                """,
                False,
                id="other",
            ),
        ],
    )
    def test_handle_matrix_item(self, content, expected_has_use_cuda_wheels):
        content, spans = parse_named_spans(content, dict)

        args = Mock()
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-use-cuda-wheels"
        )
        loader = dependencies_yaml.AnchorPreservingLoader(content)
        try:
            composed = loader.get_single_node()
        finally:
            loader.dispose()
        matrix_item_key, matrix_item = composed.value[0]

        handler = UseCUDAWheelsHandler(linter, args)
        matrix_context = Mock(
            use_cuda_wheels_node=None, has_use_cuda_wheels=False
        )
        handler.handle_matrix_item(
            matrix_context, matrix_item_key, matrix_item
        )
        expected_use_cuda_wheels_node = (
            find_yaml_node_for_span(composed, node_span)
            if (node_span := spans.get("node"))
            else None
        )
        assert matrix_context.use_cuda_wheels_node == (
            matrix_item if expected_use_cuda_wheels_node else None
        )
        assert (
            matrix_context.has_use_cuda_wheels == expected_has_use_cuda_wheels
        )

    @pytest.mark.parametrize(
        ["content"],
        [
            pytest.param(
                """\
                + matrix: {}
                : ~~~~~~original_node
                : ~~~~~~node
                + packages: []
                : ~~~~~~~~packages_key
                :           ~~packages
                """,
                id="matrix-node",
            ),
            pytest.param(
                """\
                + packages: []
                : ~~~~~~~~node
                : ~~~~~~~~packages_key
                :           ~~packages
                """,
                id="no-matrix-node",
            ),
        ],
    )
    def test_handle_packages(self, content):
        content, spans = parse_named_spans(content)

        args = Mock()
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-use-cuda-wheels"
        )
        loader = dependencies_yaml.AnchorPreservingLoader(content)
        try:
            composed = loader.get_single_node()
        finally:
            loader.dispose()
        packages_key = find_yaml_node_for_span(composed, spans["packages_key"])
        packages = find_yaml_node_for_span(composed, spans["packages"])

        original_node = (
            find_yaml_node_for_span(composed, span)
            if (span := spans.get("original_node"))
            else None
        )
        node = find_yaml_node_for_span(composed, spans["node"])

        handler = UseCUDAWheelsHandler(linter, args)
        with handler.handle_packages(
            Mock(use_cuda_wheels_node=original_node), packages_key, packages
        ) as packages_context:
            assert packages_context.use_cuda_wheels_node == node

    @pytest.mark.parametrize(
        ["content", "expected_node", "expected_name"],
        [
            pytest.param(
                "cuda-toolkit==13.0",
                True,
                "cuda-toolkit",
                id="cuda-toolkit",
            ),
            pytest.param(
                "cuda-toolkit[cufile]==13.0",
                True,
                "cuda-toolkit",
                id="cuda-toolkit-extras",
            ),
            pytest.param(
                "cupy-cuda12x[ctk]",
                True,
                "cupy-cuda12x[ctk]",
                id="cupy-ctk",
            ),
            pytest.param(
                "cupy-cuda13x[ctk,other]",
                True,
                "cupy-cuda13x[ctk]",
                id="cupy-ctk-and-other",
            ),
            pytest.param(
                "cupy-cuda13x[other]",
                False,
                None,
                id="cupy-others",
            ),
            pytest.param(
                "cupy-cuda13x",
                False,
                None,
                id="cupy-no-extras",
            ),
            pytest.param(
                "other-package",
                False,
                None,
                id="other-package",
            ),
        ],
    )
    def test_handle_package(self, content, expected_node, expected_name):
        args = Mock()
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-use-cuda-wheels"
        )
        loader = dependencies_yaml.AnchorPreservingLoader(content)
        try:
            package_node = loader.get_single_node()
        finally:
            loader.dispose()

        handler = UseCUDAWheelsHandler(linter, args)
        packages_context = Mock(suspicious_packages=[])
        handler.handle_package(packages_context, None, package_node)
        assert packages_context.suspicious_packages == (
            [(package_node, expected_name)] if expected_node else []
        )


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     common:
            :     ~~~~~~warnings.0.notes.0
            :     ~~~~~~warnings.1.notes.0
            +       - output_types: pyproject
            +         packages:
            +           - cuda-toolkit==13.0
            :             ~~~~~~~~~~~~~~~~~~warnings.0.warning
            +           - cupy-cuda13x[ctk]
            :             ~~~~~~~~~~~~~~~~~warnings.1.warning
            """,
            [
                {
                    "warning": 'package "cuda-toolkit" in common dependency '
                    "set",
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
                {
                    "warning": 'package "cupy-cuda13x[ctk]" in common '
                    "dependency set",
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
            ],
            id="common-bad-packages",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     common:
            +       - output_types: pyproject
            +         packages:
            +           - cupy-cuda13x
            +           - other-package
            """,
            [],
            id="common-no-bad-packages",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - matrix:
            +               use_cuda_wheels: "false"
            :                                ~~~~~~~warnings.0.notes.0
            :                                ~~~~~~~warnings.1.notes.0
            +             packages:
            +               - cuda-toolkit==13.0
            :                 ~~~~~~~~~~~~~~~~~~warnings.0.warning
            +               - cupy-cuda13x[ctk]
            :                 ~~~~~~~~~~~~~~~~~warnings.1.warning
            """,
            [
                {
                    "warning": 'package "cuda-toolkit" in specific dependency '
                    'set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
                {
                    "warning": 'package "cupy-cuda13x[ctk]" in specific '
                    'dependency set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
            ],
            id="specific-use-cuda-wheels-false",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - matrix:
            +               use_cuda_wheels: "true"
            +             packages:
            +               - cuda-toolkit==13.0
            +               - cupy-cuda13x[ctk]
            """,
            [],
            id="specific-use-cuda-wheels-true",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - matrix:
            +               use_cuda_wheels: "false"
            +             packages:
            +               - cupy-cuda13x
            +               - other-package
            """,
            [],
            id="specific-use-cuda-wheels-false-no-bad-packages",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - matrix:
            :             ~~~~~~warnings.0.notes.0
            :             ~~~~~~warnings.1.notes.0
            +             packages:
            +               - cuda-toolkit==13.0
            :                 ~~~~~~~~~~~~~~~~~~warnings.0.warning
            +               - cupy-cuda13x[ctk]
            :                 ~~~~~~~~~~~~~~~~~warnings.1.warning
            """,
            [
                {
                    "warning": 'package "cuda-toolkit" in specific dependency '
                    'set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
                {
                    "warning": 'package "cupy-cuda13x[ctk]" in specific '
                    'dependency set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
            ],
            id="specific-no-use-cuda-wheels",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - packages:
            +               - cuda-toolkit==13.0
            :                 ~~~~~~~~~~~~~~~~~~warnings.0.warning
            +               - cupy-cuda13x[ctk]
            :                 ~~~~~~~~~~~~~~~~~warnings.1.warning
            +             matrix:
            :             ~~~~~~warnings.0.notes.0
            :             ~~~~~~warnings.1.notes.0
            """,
            [
                {
                    "warning": 'package "cuda-toolkit" in specific dependency '
                    'set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
                {
                    "warning": 'package "cupy-cuda13x[ctk]" in specific '
                    'dependency set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
            ],
            id="specific-no-use-cuda-wheels-matrix-after-package",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - packages:
            :             ~~~~~~~~warnings.0.notes.0
            :             ~~~~~~~~warnings.1.notes.0
            +               - cuda-toolkit==13.0
            :                 ~~~~~~~~~~~~~~~~~~warnings.0.warning
            +               - cupy-cuda13x[ctk]
            :                 ~~~~~~~~~~~~~~~~~warnings.1.warning
            """,
            [
                {
                    "warning": 'package "cuda-toolkit" in specific dependency '
                    'set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
                {
                    "warning": 'package "cupy-cuda13x[ctk]" in specific '
                    'dependency set without use_cuda_wheels: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'use_cuda_wheels: "true" instead',
                    ],
                },
            ],
            id="specific-no-matrix",
        ),
    ],
)
def test_check_use_cuda_wheels_integration(content, warnings):
    content, spans = parse_named_spans(content, dict)

    args = Mock()
    linter = lint.Linter(
        "dependencies.yaml", content, "verify-use-cuda-wheels"
    )

    expected_warnings = [
        lint.LintWarning(
            warning_span["warning"],
            warning["warning"],
            notes=[
                lint.Note(note_span, note)
                for note_span, note in zip(
                    warning_span.get("notes", []),
                    warning["notes"],
                    strict=True,
                )
            ],
        )
        for warning_span, warning in zip(
            spans.get("warnings", []), warnings, strict=True
        )
    ]

    dependencies.check_dependencies(linter, args)
    assert linter.warnings == expected_warnings
