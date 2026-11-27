"""Commands and file previews must render without a Windows console handle."""
import unittest
from unittest.mock import patch
from src.tool.bash_support import get_bash_render, get_bash_result_render
from src.tool.file_io_support import fill_str_line


class TestHeadlessRendering(unittest.TestCase):
    @patch('os.get_terminal_size', side_effect=OSError('No console handle'))
    def test_bash_preview(self, _terminal):
        preview = get_bash_render('printf headless-ok')
        self.assertIn('headless-ok', preview.plain)

    @patch('os.get_terminal_size', side_effect=OSError('No console handle'))
    def test_file_line_wrap(self, _terminal):
        first, remaining = fill_str_line('a' * 250, offset=4)
        self.assertTrue(first)
        self.assertTrue(remaining)

    @patch('os.get_terminal_size', side_effect=OSError('No console handle'))
    @patch('src.tool.bash_support.BASH_RESULT_MAX_CHARS', 0)
    @patch('src.tool.bash_support.BASH_RESULT_MAX_LINES', 9)
    def test_truncation_notice_crosses_line_number_digit_boundary(self, _terminal):
        preview = get_bash_result_render('\n'.join(f'line {i}' for i in range(12)))
        self.assertIn('10', preview.plain)
        self.assertIn('(3 lines not shown)', preview.plain)


if __name__ == '__main__':
    unittest.main()
