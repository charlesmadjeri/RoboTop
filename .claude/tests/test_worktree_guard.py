import sys
import unittest
from pathlib import Path

HOOKS = Path(__file__).resolve().parent.parent / "hooks"
sys.path.insert(0, str(HOOKS))

import guard_git  # noqa: E402
from shell_text import command_text  # noqa: E402


class Heredocs(unittest.TestCase):
    def test_text_in_a_heredoc_is_not_a_command(self):
        self.assertIsNone(guard_git.check("cat > f <<'EOF'\ngit push origin dev\nEOF\n", ""))

    def test_commands_after_a_heredoc_are_still_checked(self):
        self.assertIsNotNone(guard_git.check("cat <<EOF\nx\nEOF\ngit push origin dev", ""))

    def test_a_newline_separates_commands(self):
        self.assertIsNotNone(guard_git.check("echo hi\ngit push origin dev", ""))

    def test_a_heredoc_inside_quotes_is_kept(self):
        command = "git commit -s -m \"$(cat <<'EOF'\nfix(api): x\nEOF\n)\""
        self.assertEqual(command_text(command), command)


if __name__ == "__main__":
    unittest.main()
