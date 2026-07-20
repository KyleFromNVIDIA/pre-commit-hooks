# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os.path
from itertools import chain
from unittest.mock import Mock, patch

import pytest
from packaging.version import Version
from rapids_metadata.metadata import (
    RAPIDSMetadata,
    RAPIDSRepository,
    RAPIDSVersion,
)

from rapids_pre_commit_hooks import alpha_spec, lint
from rapids_pre_commit_hooks.utils import dependencies_yaml
from rapids_pre_commit_hooks_test_utils import parse_named_spans

latest_version, latest_metadata = max(
    alpha_spec.all_metadata().versions.items(),
    key=lambda item: Version(item[0]),
)


@contextlib.contextmanager
def set_cwd(cwd):
    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@pytest.mark.parametrize(
    [
        "version_file",
        "version_file_contents",
        "version_arg",
        "expected_version",
        "raises",
    ],
    [
        ("VERSION", "24.06", None, "24.06", contextlib.nullcontext()),
        ("RAPIDS_VERSION", "24.06", None, "24.06", contextlib.nullcontext()),
        ("VERSION", "24.06", "24.08", "24.08", contextlib.nullcontext()),
        ("VERSION", "24.08", "24.06", "24.06", contextlib.nullcontext()),
        ("VERSION", None, "24.06", "24.06", contextlib.nullcontext()),
        ("VERSION", None, "24.10", None, pytest.raises(KeyError)),
        ("VERSION", None, None, None, pytest.raises(FileNotFoundError)),
    ],
)
def test_get_rapids_version(
    tmp_path,
    version_file,
    version_file_contents,
    version_arg,
    expected_version,
    raises,
):
    MOCK_METADATA = RAPIDSMetadata(
        versions={
            "24.06": RAPIDSVersion(
                repositories={
                    "repo1": RAPIDSRepository(),
                },
            ),
            "24.08": RAPIDSVersion(
                repositories={
                    "repo2": RAPIDSRepository(),
                },
            ),
        },
    )
    with (
        set_cwd(tmp_path),
        patch(
            "rapids_pre_commit_hooks.alpha_spec.all_metadata",
            Mock(return_value=MOCK_METADATA),
        ),
    ):
        if version_file_contents:
            with open(version_file, "w") as f:
                f.write(f"{version_file_contents}\n")
        args = Mock(
            rapids_version=version_arg, rapids_version_file=version_file
        )
        with raises:
            version = alpha_spec.get_rapids_version(args)
            if expected_version:
                assert version == MOCK_METADATA.versions[expected_version]


@pytest.mark.parametrize(
    ["name", "stripped_name"],
    [
        *chain(
            *(
                [
                    (p, p),
                    (f"{p}-cu11", p),
                    (f"{p}-cu12", p),
                    (f"{p}-cuda", f"{p}-cuda"),
                ]
                for p in latest_metadata.cuda_suffixed_packages
            )
        ),
        *chain(
            *(
                [
                    (p, p),
                    (f"{p}-cu11", f"{p}-cu11"),
                    (f"{p}-cu12", f"{p}-cu12"),
                    (f"{p}-cuda", f"{p}-cuda"),
                ]
                for p in latest_metadata.all_packages
                - latest_metadata.cuda_suffixed_packages
            )
        ),
    ],
)
@patch(
    "rapids_pre_commit_hooks.alpha_spec.get_rapids_version",
    Mock(return_value=latest_metadata),
)
def test_strip_cuda_suffix(name, stripped_name):
    assert alpha_spec.strip_cuda_suffix(Mock(), name) == stripped_name


@pytest.mark.parametrize(
    ["package", "anchor", "content", "mode", "replacement"],
    [
        *chain(
            *(
                [
                    (p, None, p, "development", f"{p}>=0.0.0a0"),
                    (p, None, p, "release", None),
                    (p, None, f"{p}>=0.0.0a0", "development", None),
                    (p, None, f"{p}>=0.0.0a0", "release", p),
                ]
                for p in latest_metadata.prerelease_packages
            )
        ),
        *chain(
            *(
                [
                    (
                        f"{p}-cu12",
                        None,
                        f"{p}-cu12",
                        "development",
                        f"{p}-cu12>=0.0.0a0",
                    ),
                    (f"{p}-cu11", None, f"{p}-cu11", "release", None),
                    (
                        f"{p}-cu12",
                        None,
                        f"{p}-cu12>=0.0.0a0",
                        "development",
                        None,
                    ),
                    (
                        f"{p}-cu11",
                        None,
                        f"{p}-cu11>=0.0.0a0",
                        "release",
                        f"{p}-cu11",
                    ),
                ]
                for p in latest_metadata.prerelease_packages
                & latest_metadata.cuda_suffixed_packages
            )
        ),
        *chain(
            *(
                [
                    (f"{p}-cu12", None, f"{p}-cu12", "development", None),
                    (f"{p}-cu12", None, f"{p}-cu12>=0.0.0a0", "release", None),
                ]
                for p in latest_metadata.prerelease_packages
                & (
                    latest_metadata.all_packages
                    - latest_metadata.cuda_suffixed_packages
                )
            )
        ),
        (
            "cuml",
            None,
            "cuml>=24.04,<24.06",
            "development",
            "cuml>=24.04,<24.06,>=0.0.0a0",
        ),
        (
            "cuml",
            None,
            "cuml>=24.04,<24.06,>=0.0.0a0",
            "release",
            "cuml>=24.04,<24.06",
        ),
        (
            "cuml",
            "cuml",
            "&cuml cuml>=24.04,<24.06,>=0.0.0a0",
            "release",
            "&cuml cuml>=24.04,<24.06",
        ),
        ("packaging", None, "packaging", "development", None),
        (
            None,
            None,
            "--extra-index-url=https://pypi.nvidia.com",
            "development",
            None,
        ),
        (
            None,
            None,
            "--extra-index-url=https://pypi.nvidia.com",
            "release",
            None,
        ),
        (None, None, "gcc_linux-64=11.*", "development", None),
        (None, None, "gcc_linux-64=11.*", "release", None),
    ],
)
@patch(
    "rapids_pre_commit_hooks.alpha_spec.get_rapids_version",
    Mock(return_value=latest_metadata),
)
def test_check_package_spec(package, anchor, content, mode, replacement):
    args = Mock(mode=mode)
    linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
    loader = dependencies_yaml.AnchorPreservingLoader(content)
    try:
        composed = loader.get_single_node()
    finally:
        loader.dispose()
    handler = alpha_spec.AlphaSpecHandler(linter, args)
    handler.handle_package(Mock(), anchor, composed)
    if replacement is None:
        assert linter.warnings == []
    else:
        expected_linter = lint.Linter(
            "dependencies.yaml", content, "verify-alpha-spec"
        )
        expected_linter.add_warning(
            (composed.start_mark.index, composed.end_mark.index),
            f"{'add' if mode == 'development' else 'remove'} "
            f"alpha spec for RAPIDS package {package}",
        ).add_replacement(
            (composed.start_mark.index, composed.end_mark.index), replacement
        )
        assert linter.warnings == expected_linter.warnings


def test_check_alpha_spec():
    CONTENT = "dependencies: []"
    with (
        patch(
            "rapids_pre_commit_hooks.alpha_spec.AlphaSpecHandler", Mock()
        ) as mock_alpha_spec_handler,
        patch(
            "rapids_pre_commit_hooks.alpha_spec.traverse_dependencies_yaml",
            Mock(),
        ) as mock_traverse_dependencies_yaml,
    ):
        args = Mock()
        linter = lint.Linter("dependencies.yaml", CONTENT, "verify-alpha-spec")
        alpha_spec.check_alpha_spec(linter, args)
    mock_alpha_spec_handler.assert_called_once()
    mock_traverse_dependencies_yaml.assert_called_once_with(
        mock_alpha_spec_handler(), CONTENT
    )


def test_check_alpha_spec_integration(tmp_path):
    content, spans = parse_named_spans(
        """\
        + dependencies:
        +   test:
        +     common:
        +       - output_types: pyproject
        +         packages:
        +           - cudf>=24.04,<24.06
        :             ~~~~~~~~~~~~~~~~~~package
        """
    )

    args = Mock(
        mode="development", rapids_version=None, rapids_version_file="VERSION"
    )
    linter = lint.Linter("dependencies.yaml", content, "verify-alpha-spec")
    with open(os.path.join(tmp_path, "VERSION"), "w") as f:
        f.write(f"{latest_version}\n")
    with set_cwd(tmp_path):
        alpha_spec.check_alpha_spec(linter, args)

    expected_linter = lint.Linter(
        "dependencies.yaml", content, "verify-alpha-spec"
    )
    expected_linter.add_warning(
        spans["package"], "add alpha spec for RAPIDS package cudf"
    ).add_replacement(spans["package"], "cudf>=24.04,<24.06,>=0.0.0a0")
    assert linter.warnings == expected_linter.warnings
