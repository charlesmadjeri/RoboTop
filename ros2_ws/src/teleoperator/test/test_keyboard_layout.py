import subprocess
from types import SimpleNamespace

import pytest
from teleoperator.keyboard_layout import (
    AXIS_KEYS,
    describe_keys,
    is_shifted,
    key_name,
    resolve_layout,
    system_layout_name,
)


def test_a_french_system_layout_selects_azerty():
    assert resolve_layout("auto", lambda: "fr")[0] == "azerty"


def test_a_us_system_layout_selects_qwerty():
    assert resolve_layout("auto", lambda: "us")[0] == "qwerty"


def test_only_the_first_listed_layout_counts_in_a_multi_layout_setup(monkeypatch):
    output = "rules:      evdev\nlayout:     us,fr\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
    assert system_layout_name() == "us"


def test_the_environment_layout_is_used_when_no_tool_answers(monkeypatch):
    def missing_tool(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_tool)
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", "fr")
    assert system_layout_name() == "fr"


def test_a_german_system_layout_selects_qwertz():
    assert resolve_layout("auto", lambda: "de")[0] == "qwertz"


def test_the_parameter_overrides_the_detection():
    assert resolve_layout("azerty", lambda: "us") == ("azerty", "parameter")


def test_an_unreadable_system_layout_falls_back_to_azerty():
    layout, origin = resolve_layout("auto", lambda: "")
    assert layout == "azerty"
    assert "could not be read" in origin


@pytest.mark.parametrize("layout", sorted(AXIS_KEYS))
def test_every_layout_drives_the_three_axes_in_both_directions(layout):
    assert sorted(AXIS_KEYS[layout].values()) == [(0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)]


def test_qwerty_and_azerty_use_the_same_physical_keys():
    qwerty, azerty = AXIS_KEYS["qwerty"], AXIS_KEYS["azerty"]
    assert qwerty["w"] == azerty["z"] and qwerty["a"] == azerty["q"] and qwerty[";"] == azerty["m"]


def test_shifted_characters_name_the_same_key_and_count_as_boost():
    assert key_name("W") == "w" and key_name(":") == ";"
    assert is_shifted("W") and is_shifted(":") and not is_shifted("w")


def test_the_key_description_lists_forward_before_backward():
    assert describe_keys("qwerty") == "x: W/S, y: A/D, z: P/;"
