# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, Mock, call, patch

import pytest
import yaml
from rapids_pre_commit_hooks.utils import dependencies_yaml


def test_anchor_preserving_loader():
    loader = dependencies_yaml.AnchorPreservingLoader("- &a A\n- *a")
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
    actual_descend, actual_anchor = dependencies_yaml.check_and_mark_anchor(
        ANCHORS, used_anchors, NODES[node_index]
    )
    assert actual_descend == descend
    assert actual_anchor == anchor
    assert used_anchors == used_anchors_after


def test_traverse_package():
    packages = yaml.SafeLoader("""\
    - lib1
    """).get_single_node()
    package = packages.value[0]
    packages_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_package(packages_context, None, package),
    ]
    manager.reset_mock()

    dependencies_yaml.traverse_package(
        manager.handler, packages_context, {}, set(), package
    )

    assert manager.mock_calls == expected_calls


def test_traverse_package_anchor():
    packages = yaml.SafeLoader("""\
    - &lib1 lib1
    - *lib1
    """).get_single_node()
    package = packages.value[0]
    packages_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_package(packages_context, "lib1", package),
    ]
    manager.reset_mock()

    dependencies_yaml.traverse_package(
        manager.handler, packages_context, {"lib1": package}, set(), package
    )

    assert manager.mock_calls == expected_calls


def test_traverse_package_used_anchor():
    packages = yaml.SafeLoader("""\
    - &lib1 lib1
    - *lib1
    """).get_single_node()
    package = packages.value[1]
    packages_context = Mock()
    manager = MagicMock()

    expected_calls = []
    manager.reset_mock()

    dependencies_yaml.traverse_package(
        manager.handler, packages_context, {"lib1": package}, {"lib1"}, package
    )

    assert manager.mock_calls == expected_calls


def test_traverse_packages():
    item = yaml.SafeLoader("""\
    packages:
        - lib1
        - lib2
    """).get_single_node()
    packages_key, packages = item.value[0]
    item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_packages(item_context, packages_key, packages),
        call.handler.handle_packages().__enter__(),
        call.traverse_package(
            manager.handler,
            manager.handler.handle_packages().__enter__(),
            {},
            set(),
            packages.value[0],
        ),
        call.traverse_package(
            manager.handler,
            manager.handler.handle_packages().__enter__(),
            {},
            set(),
            packages.value[1],
        ),
        call.handler.handle_packages().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_package",
            manager.traverse_package,
        ),
    ):
        dependencies_yaml.traverse_packages(
            manager.handler, item_context, {}, set(), packages_key, packages
        )

    assert manager.mock_calls == expected_calls


def test_traverse_packages_anchor():
    items = yaml.SafeLoader("""\
    - packages: &packages
        - lib1
        - lib2
    - packages: *packages
    """).get_single_node()
    packages_key, packages = items.value[0].value[0]
    item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_packages(item_context, packages_key, packages),
        call.handler.handle_packages().__enter__(),
        call.traverse_package(
            manager.handler,
            manager.handler.handle_packages().__enter__(),
            {"packages": items.value[0].value[0][1]},
            {"packages"},
            packages.value[0],
        ),
        call.traverse_package(
            manager.handler,
            manager.handler.handle_packages().__enter__(),
            {"packages": items.value[0].value[0][1]},
            {"packages"},
            packages.value[1],
        ),
        call.handler.handle_packages().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_package",
            manager.traverse_package,
        ),
    ):
        dependencies_yaml.traverse_packages(
            manager.handler,
            item_context,
            {"packages": items.value[0].value[0][1]},
            set(),
            packages_key,
            packages,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_packages_used_anchor():
    items = yaml.SafeLoader("""\
    - packages: &packages
        - lib1
        - lib2
    - packages: *packages
    """).get_single_node()
    packages_key, packages = items.value[1].value[0]
    item_context = Mock()
    manager = MagicMock()

    expected_calls = []
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_package",
            manager.traverse_package,
        ),
    ):
        dependencies_yaml.traverse_packages(
            manager.handler,
            item_context,
            {"packages": items.value[0].value[0][1]},
            {"packages"},
            packages_key,
            packages,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_common_item():
    common = yaml.SafeLoader("""\
    - output_types: pyproject
      packages: []
    """).get_single_node()
    common_item = common.value[0]
    common_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_common_item(common_context, common_item),
        call.handler.handle_common_item().__enter__(),
        call.traverse_packages(
            manager.handler,
            manager.handler.handle_common_item().__enter__(),
            {},
            set(),
            common_item.value[1][0],
            common_item.value[1][1],
        ),
        call.handler.handle_common_item().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_packages",
            manager.traverse_packages,
        ),
    ):
        dependencies_yaml.traverse_common_item(
            manager.handler, common_context, {}, set(), common_item
        )

    assert manager.mock_calls == expected_calls


def test_traverse_common():
    dependency_set = yaml.SafeLoader("""\
    common:
        - {}
        - {}
    """).get_single_node()
    common_key, common = dependency_set.value[0]
    dependency_set_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_common(dependency_set_context, common_key, common),
        call.handler.handle_common().__enter__(),
        call.traverse_common_item(
            manager.handler,
            manager.handler.handle_common().__enter__(),
            {},
            set(),
            common.value[0],
        ),
        call.traverse_common_item(
            manager.handler,
            manager.handler.handle_common().__enter__(),
            {},
            set(),
            common.value[1],
        ),
        call.handler.handle_common().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_common_item",
            manager.traverse_common_item,
        ),
    ):
        dependencies_yaml.traverse_common(
            manager.handler,
            dependency_set_context,
            {},
            set(),
            common_key,
            common,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_matrix_item():
    matrix = yaml.SafeLoader("""\
    value_1: "true"
    """).get_single_node()
    matrix_item_key, matrix_item = matrix.value[0]
    matrix_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrix_item(
            matrix_context, matrix_item_key, matrix_item
        ),
    ]
    manager.reset_mock()

    dependencies_yaml.traverse_matrix_item(
        manager.handler,
        matrix_context,
        matrix_item_key,
        matrix_item,
    )

    assert manager.mock_calls == expected_calls


def test_traverse_matrix():
    matrices_item = yaml.SafeLoader("""\
    matrix:
        value_1: "true"
        value_2: "true"
    """).get_single_node()
    matrix_key, matrix = matrices_item.value[0]
    matrices_item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrix(matrices_item_context, matrix_key, matrix),
        call.handler.handle_matrix().__enter__(),
        call.traverse_matrix_item(
            manager.handler,
            manager.handler.handle_matrix().__enter__(),
            matrix.value[0][0],
            matrix.value[0][1],
        ),
        call.traverse_matrix_item(
            manager.handler,
            manager.handler.handle_matrix().__enter__(),
            matrix.value[1][0],
            matrix.value[1][1],
        ),
        call.handler.handle_matrix().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrix_item",
            manager.traverse_matrix_item,
        ),
    ):
        dependencies_yaml.traverse_matrix(
            manager.handler,
            matrices_item_context,
            matrix_key,
            matrix,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_matrices_item():
    matrices = yaml.SafeLoader("""\
    - matrix: {}
      packages: []
    """).get_single_node()
    matrices_item = matrices.value[0]
    matrices_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrices_item(matrices_context, matrices_item),
        call.handler.handle_matrices_item().__enter__(),
        call.traverse_matrix(
            manager.handler,
            manager.handler.handle_matrices_item().__enter__(),
            matrices_item.value[0][0],
            matrices_item.value[0][1],
        ),
        call.traverse_packages(
            manager.handler,
            manager.handler.handle_matrices_item().__enter__(),
            {},
            set(),
            matrices_item.value[1][0],
            matrices_item.value[1][1],
        ),
        call.handler.handle_matrices_item().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrix",
            manager.traverse_matrix,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_packages",
            manager.traverse_packages,
        ),
    ):
        dependencies_yaml.traverse_matrices_item(
            manager.handler, matrices_context, {}, set(), matrices_item
        )

    assert manager.mock_calls == expected_calls


def test_traverse_matrices():
    specific_item = yaml.SafeLoader("""\
    matrices:
        - {}
        - {}
        - {}
    """).get_single_node()
    matrices_key, matrices = specific_item.value[0]
    specific_item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrices(
            specific_item_context, matrices_key, matrices
        ),
        call.handler.handle_matrices().__enter__(),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            matrices.value[0],
        ),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            matrices.value[1],
        ),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            matrices.value[2],
        ),
        call.handler.handle_matrices().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrices_item",
            manager.traverse_matrices_item,
        ),
    ):
        dependencies_yaml.traverse_matrices(
            manager.handler,
            specific_item_context,
            {},
            set(),
            matrices_key,
            matrices,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_specific_item():
    specific = yaml.SafeLoader("""\
    - output_types: pyproject
      matrices: []
    """).get_single_node()
    specific_item = specific.value[0]
    specific_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_specific_item(specific_context, specific_item),
        call.handler.handle_specific_item().__enter__(),
        call.traverse_matrices(
            manager.handler,
            manager.handler.handle_specific_item().__enter__(),
            {},
            set(),
            specific_item.value[1][0],
            specific_item.value[1][1],
        ),
        call.handler.handle_specific_item().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrices",
            manager.traverse_matrices,
        ),
    ):
        dependencies_yaml.traverse_specific_item(
            manager.handler, specific_context, {}, set(), specific_item
        )

    assert manager.mock_calls == expected_calls


def test_traverse_specific():
    dependency_set = yaml.SafeLoader("""\
    specific:
        - {}
        - {}
        - {}
    """).get_single_node()
    specific_key, specific = dependency_set.value[0]
    dependency_set_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_specific(
            dependency_set_context, specific_key, specific
        ),
        call.handler.handle_specific().__enter__(),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            specific.value[0],
        ),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            specific.value[1],
        ),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            specific.value[2],
        ),
        call.handler.handle_specific().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_specific_item",
            manager.traverse_specific_item,
        ),
    ):
        dependencies_yaml.traverse_specific(
            manager.handler,
            dependency_set_context,
            {},
            set(),
            specific_key,
            specific,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_dependency_set():
    dependencies = yaml.SafeLoader("""\
    dependency_set_1:
        common: {}
        specific: {}
    """).get_single_node()
    dependency_set_key, dependency_set = dependencies.value[0]
    dependencies_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_dependency_set(
            dependencies_context, dependency_set_key, dependency_set
        ),
        call.handler.handle_dependency_set().__enter__(),
        call.traverse_common(
            manager.handler,
            manager.handler.handle_dependency_set().__enter__(),
            {},
            set(),
            dependency_set.value[0][0],
            dependency_set.value[0][1],
        ),
        call.traverse_specific(
            manager.handler,
            manager.handler.handle_dependency_set().__enter__(),
            {},
            set(),
            dependency_set.value[1][0],
            dependency_set.value[1][1],
        ),
        call.handler.handle_dependency_set().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_common",
            manager.traverse_common,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_specific",
            manager.traverse_specific,
        ),
    ):
        dependencies_yaml.traverse_dependency_set(
            manager.handler,
            dependencies_context,
            {},
            set(),
            dependency_set_key,
            dependency_set,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_dependencies():
    root = yaml.SafeLoader("""\
    dependencies:
        dependency_set_1: {}
        dependency_set_2: {}
    """).get_single_node()
    dependencies_key, dependencies = root.value[0]
    root_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_dependencies(
            root_context, dependencies_key, dependencies
        ),
        call.handler.handle_dependencies().__enter__(),
        call.traverse_dependency_set(
            manager.handler,
            manager.handler.handle_dependencies().__enter__(),
            {},
            set(),
            dependencies.value[0][0],
            dependencies.value[0][1],
        ),
        call.traverse_dependency_set(
            manager.handler,
            manager.handler.handle_dependencies().__enter__(),
            {},
            set(),
            dependencies.value[1][0],
            dependencies.value[1][1],
        ),
        call.handler.handle_dependencies().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with patch(
        "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_dependency_set",
        manager.traverse_dependency_set,
    ):
        dependencies_yaml.traverse_dependencies(
            manager.handler,
            root_context,
            {},
            set(),
            dependencies_key,
            dependencies,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_root():
    root = yaml.SafeLoader("""\
    files: {}
    channels: []
    dependencies: {}
    """).get_single_node()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_root(root),
        call.handler.handle_root().__enter__(),
        call.traverse_dependencies(
            manager.handler,
            manager.handler.handle_root().__enter__(),
            {},
            set(),
            root.value[2][0],
            root.value[2][1],
        ),
        call.handler.handle_root().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with patch(
        "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_dependencies",
        manager.traverse_dependencies,
    ):
        dependencies_yaml.traverse_root(manager.handler, {}, set(), root)

    assert manager.mock_calls == expected_calls
