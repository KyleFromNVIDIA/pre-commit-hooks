# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
from textwrap import dedent
from unittest.mock import Mock, call, patch

import pytest
import yaml

from rapids_pre_commit_hooks import alpha_spec, lint


@pytest.mark.parametrize(
    ["package", "content", "mode", "replacement"],
    [
        *chain(
            *(
                [
                    (p, p, "development", f"{p}>=0.0.0a0"),
                    (p, p, "release", None),
                    (p, f"{p}>=0.0.0a0", "development", None),
                    (p, f"{p}>=0.0.0a0", "release", p),
                ]
                for p in alpha_spec.RAPIDS_VERSIONED_PACKAGES
            )
        ),
        ("cuml", "cuml>=24.04,<=24.06", "development", "cuml<=24.06,>=0.0.0a0,>=24.04"),
        ("cuml", "cuml>=24.04,<=24.06,>=0.0.0a0", "release", "cuml<=24.06,>=24.04"),
        ("packaging", "packaging", "development", None),
    ],
)
def test_check_package_spec(package, content, mode, replacement):
    args = Mock(mode=mode)
    linter = lint.Linter("dependencies.yaml", content)
    composed = yaml.compose(content)
    alpha_spec.check_package_spec(linter, args, composed)
    if replacement is None:
        assert linter.warnings == []
    else:
        expected_linter = lint.Linter("dependencies.yaml", content)
        expected_linter.add_warning(
            (composed.start_mark.index, composed.end_mark.index),
            f"{'add' if mode == 'development' else 'remove'} "
            f"alpha spec for RAPIDS package {package}",
        ).add_replacement((0, len(content)), replacement)
        assert linter.warnings == expected_linter.warnings


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - package_a
                - package_b
                """
            ),
            [0, 1],
        ),
        (
            "null",
            [],
        ),
    ],
)
def test_check_packages(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_package_spec", Mock()
    ) as mock_check_package_spec:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        composed = yaml.compose(content)
        alpha_spec.check_packages(linter, args, composed)
    assert mock_check_package_spec.mock_calls == [
        call(linter, args, composed.value[i]) for i in indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - output_types: [pyproject, conda]
                  packages:
                    - package_a
                - output_types: [conda]
                  packages:
                    - package_b
                - packages:
                    - package_c
                  output_types: pyproject
                """
            ),
            [(0, 1), (2, 0)],
        ),
    ],
)
def test_check_common(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_packages", Mock()
    ) as mock_check_packages:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        composed = yaml.compose(content)
        alpha_spec.check_common(linter, args, composed)
    assert mock_check_packages.mock_calls == [
        call(linter, args, composed.value[i].value[j][1]) for i, j in indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - matrix:
                    arch: x86_64
                  packages:
                    - package_a
                - packages:
                    - package_b
                  matrix:
                """
            ),
            [(0, 1), (1, 0)],
        ),
    ],
)
def test_check_matrices(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_packages", Mock()
    ) as mock_check_packages:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        composed = yaml.compose(content)
        alpha_spec.check_matrices(linter, args, composed)
    assert mock_check_packages.mock_calls == [
        call(linter, args, composed.value[i].value[j][1]) for i, j in indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
                - output_types: [pyproject, conda]
                  matrices:
                    - matrix:
                        arch: x86_64
                      packages:
                        - package_a
                - output_types: [conda]
                  matrices:
                    - matrix:
                        arch: x86_64
                      packages:
                        - package_b
                - matrices:
                    - matrix:
                        arch: x86_64
                      packages:
                        - package_c
                  output_types: pyproject
                """
            ),
            [(0, 1), (2, 0)],
        ),
    ],
)
def test_check_specific(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_matrices", Mock()
    ) as mock_check_matrices:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        composed = yaml.compose(content)
        alpha_spec.check_specific(linter, args, composed)
    assert mock_check_matrices.mock_calls == [
        call(linter, args, composed.value[i].value[j][1]) for i, j in indices
    ]


@pytest.mark.parametrize(
    ["content", "common_indices", "specific_indices"],
    [
        (
            dedent(
                """\
                set_a:
                  common:
                    - output_types: [pyproject]
                      packages:
                        - package_a
                  specific:
                    - output_types: [pyproject]
                      matrices:
                        - matrix:
                            arch: x86_64
                          packages:
                            - package_b
                set_b:
                  specific:
                    - output_types: [pyproject]
                      matrices:
                        - matrix:
                            arch: x86_64
                          packages:
                            - package_c
                  common:
                    - output_types: [pyproject]
                      packages:
                        - package_d
                """
            ),
            [(0, 0), (1, 1)],
            [(0, 1), (1, 0)],
        ),
    ],
)
def test_check_dependencies(content, common_indices, specific_indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_common", Mock()
    ) as mock_check_common, patch(
        "rapids_pre_commit_hooks.alpha_spec.check_specific", Mock()
    ) as mock_check_specific:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        composed = yaml.compose(content)
        alpha_spec.check_dependencies(linter, args, composed)
    assert mock_check_common.mock_calls == [
        call(linter, args, composed.value[i][1].value[j][1]) for i, j in common_indices
    ]
    assert mock_check_specific.mock_calls == [
        call(linter, args, composed.value[i][1].value[j][1])
        for i, j in specific_indices
    ]


@pytest.mark.parametrize(
    ["content", "indices"],
    [
        (
            dedent(
                """\
            files: {}
            channels: []
            dependencies: {}
            """
            ),
            [2],
        ),
    ],
)
def test_check_root(content, indices):
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_dependencies", Mock()
    ) as mock_check_dependencies:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", content)
        composed = yaml.compose(content)
        alpha_spec.check_root(linter, args, composed)
    assert mock_check_dependencies.mock_calls == [
        call(linter, args, composed.value[i][1]) for i in indices
    ]


def test_check_alpha_spec():
    CONTENT = "dependencies: []"
    with patch(
        "rapids_pre_commit_hooks.alpha_spec.check_root", Mock()
    ) as mock_check_root, patch("yaml.compose", Mock()) as mock_yaml_compose:
        args = Mock()
        linter = lint.Linter("dependencies.yaml", CONTENT)
        alpha_spec.check_alpha_spec(linter, args)
    mock_yaml_compose.assert_called_once_with(CONTENT)
    mock_check_root.assert_called_once_with(linter, args, mock_yaml_compose())
