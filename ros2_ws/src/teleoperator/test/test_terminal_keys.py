from teleoperator.keyboard_layout import AXIS_KEYS
from teleoperator.terminal_keys import TerminalKeys


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def keys(clock):
    return TerminalKeys(AXIS_KEYS["qwerty"], hold_s=0.1, clock=clock)


def test_a_pressed_key_stays_held_for_the_hold_window_then_releases():
    clock = FakeClock()
    source = keys(clock)
    source.feed("w")
    assert source.snapshot() == ({"w"}, False)
    clock.now = 0.2
    assert source.snapshot() == (set(), False)


def test_repeated_characters_keep_the_key_held():
    clock = FakeClock()
    source = keys(clock)
    for tick in range(5):
        clock.now = tick * 0.05
        source.feed("w")
    assert source.snapshot()[0] == {"w"}


def test_an_uppercase_letter_is_the_same_key_with_boost():
    source = keys(FakeClock())
    source.feed("D")
    assert source.snapshot() == ({"d"}, True)


def test_the_shifted_symbol_of_the_last_qwerty_key_is_boosted_too():
    source = keys(FakeClock())
    source.feed(":")
    assert source.snapshot() == ({";"}, True)


def test_characters_outside_the_layout_are_ignored():
    source = keys(FakeClock())
    source.feed_text("zq")
    assert source.snapshot() == (set(), False)


def test_an_arrow_key_is_not_read_as_a_boosted_letter():
    source = keys(FakeClock())
    source.feed_text("\x1b[A\x1b[1;2D\x1bOP")
    assert source.snapshot() == (set(), False)


def test_a_letter_typed_right_after_an_arrow_key_still_counts():
    source = keys(FakeClock())
    source.feed_text("\x1b[Aw")
    assert source.snapshot() == ({"w"}, False)


def test_two_keys_are_held_at_once():
    source = keys(FakeClock())
    source.feed("w")
    source.feed("a")
    assert source.snapshot()[0] == {"w", "a"}
