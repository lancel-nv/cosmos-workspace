import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "rewindow", Path(__file__).parents[1] / "tools/rewindow_robodojo_bimanual.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("actions", [16, 32, 99, 238, 256, 444, 592, 600, 1200])
def test_stride_one_preserves_every_action_and_one_terminal(actions):
    windows = module.windows_for(actions, 592)
    covered = {index for start, end in windows for index in range(start, end)}
    assert covered == set(range(actions))
    assert sum(end == actions for start, end in windows) == 1
    assert [start for start, end in windows] == list(range(len(windows)))
    assert all(16 <= end - start <= 592 and (end - start) % 16 == 0 for start, end in windows)


@pytest.mark.parametrize("actions,maximum", [(15, 592), (600, 593), (600, 0)])
def test_invalid_lengths_fail(actions, maximum):
    with pytest.raises(ValueError):
        module.windows_for(actions, maximum)
