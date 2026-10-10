"""Offline regression checks for the downloader; no ECG requests or training.

Run: python -B -m unittest discover -s tests -p test_prepare_dataset_1000_download.py -v
The download function is loaded from its actual source with standard-library
imports only, so these checks do not require NumPy, pandas, or WFDB.
"""

import ast
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def load_downloader():
    source_path = Path(__file__).resolve().parents[1] / "prepare_dataset_1000.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    compile(tree, str(source_path), "exec")  # Also check the complete script's syntax.
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import) and all(
            alias.name not in {"numpy", "pandas", "wfdb"} for alias in node.names
        ):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"DOWNLOAD_TIMEOUT_SECONDS", "DOWNLOAD_MAX_SECONDS"}
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "download_with_retries":
            nodes.append(node)
    module = types.ModuleType("downloader_under_test")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), module.__dict__)
    return module


DOWNLOADER = load_downloader()


def response_for(chunks, content_length=None):
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {} if content_length is None else {"Content-Length": str(content_length)}
    response.read1.side_effect = chunks
    return response


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "sample.hea"

    def run_download(self, **options):
        with patch.object(DOWNLOADER.time, "sleep"), patch("builtins.print"):
            return DOWNLOADER.download_with_retries(
                "https://example.invalid/sample.hea", self.destination, 3, **options
            )

    def test_existing_cache_is_preserved_without_http_request(self):
        self.destination.write_bytes(b"existing valid file")
        with patch.object(DOWNLOADER, "urlopen") as request:
            self.assertEqual(self.run_download(), "reused")
        request.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"existing valid file")

    def test_connection_timeout_stops_after_three_attempts(self):
        with patch.object(DOWNLOADER, "urlopen", side_effect=TimeoutError("silent server")) as request:
            with self.assertRaisesRegex(RuntimeError, "after_3_attempts"):
                self.run_download(timeout_seconds=7)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args.kwargs["timeout"], 7)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_read_timeout_retries_then_publishes_complete_file(self):
        failed = response_for([b"partial", TimeoutError("read stalled")], 20)
        succeeded = response_for([b"complete", b""], 8)
        with patch.object(DOWNLOADER, "urlopen", side_effect=[failed, succeeded]) as request:
            self.assertEqual(self.run_download(), "downloaded")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.destination.read_bytes(), b"complete")
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_short_content_is_never_published_as_cache(self):
        responses = [response_for([b"short", b""], 100) for _ in range(3)]
        with patch.object(DOWNLOADER, "urlopen", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "Incomplete download"):
                self.run_download()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_empty_response_is_rejected(self):
        responses = [response_for([b""]) for _ in range(3)]
        with patch.object(DOWNLOADER, "urlopen", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "Empty download"):
                self.run_download()
        self.assertFalse(self.destination.exists())

    def test_earlier_partial_file_is_not_reused_or_removed(self):
        earlier_partial = self.root / ".sample.hea.earlier.part"
        earlier_partial.write_bytes(b"unfinished earlier attempt")
        with patch.object(DOWNLOADER, "urlopen", return_value=response_for([b"new", b""], 3)):
            self.assertEqual(self.run_download(), "downloaded")
        self.assertEqual(earlier_partial.read_bytes(), b"unfinished earlier attempt")
        self.assertEqual(self.destination.read_bytes(), b"new")

    def test_keyboard_interrupt_cleans_only_current_partial(self):
        interrupted = response_for([b"partial", KeyboardInterrupt()])
        with patch.object(DOWNLOADER, "urlopen", return_value=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_download()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_elapsed_limit_retries_without_publishing_partial(self):
        responses = [response_for([b"slow"]) for _ in range(3)]
        # started -> before read -> after read exceeds the elapsed limit.
        with patch.object(DOWNLOADER, "urlopen", side_effect=responses) as request:
            with patch.object(DOWNLOADER.time, "monotonic", side_effect=[0, 0, 121] * 3):
                with self.assertRaisesRegex(RuntimeError, "elapsed limit"):
                    self.run_download(max_download_seconds=120)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_empty_existing_cache_is_preserved_and_reported(self):
        self.destination.write_bytes(b"")
        with patch.object(DOWNLOADER, "urlopen") as request:
            with self.assertRaisesRegex(RuntimeError, "invalid_existing_cache_preserved"):
                self.run_download()
        request.assert_not_called()
        self.assertTrue(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
