#!/usr/bin/env python3
"""Regression checks for protection failures, using only disposable PNG fixtures."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import verify_frame_protection as verifier


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract_path = self.root / "contract.json"
        self.save("baseline.png", [(10, 20, 30, 255), (40, 50, 60, 0)])
        self.save("candidate.png", [(70, 80, 90, 255), (40, 50, 60, 0)])
        # Alpha is opaque at BOTH pixels; only the first red byte permits edits.
        self.save("mask.png", [(255, 255, 255, 255), (0, 0, 0, 255)])
        self.contract = {"edits": [{
            "baseline": "baseline.png", "baseline_sha256": self.sha("baseline.png"),
            "candidate": "candidate.png", "candidate_sha256": self.sha("candidate.png"),
            "allowed_mask": "mask.png", "allowed_mask_sha256": self.sha("mask.png"),
            "mask_origin": "Fixture chosen before comparison; first pixel only",
        }]}

    def save(self, filename, pixels, size=(2, 1), mode="RGBA"):
        image = Image.new(mode, size)
        image.putdata(pixels)
        image.save(self.root / filename)

    def sha(self, filename):
        return hashlib.sha256((self.root / filename).read_bytes()).hexdigest()

    def run_contract(self, *extra):
        self.contract_path.write_text(json.dumps(self.contract))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = verifier.main([str(self.contract_path), *map(str, extra)])
        return code, json.loads(output.getvalue())

    def test_local_edit_uses_red_not_opaque_alpha_and_preserves_input_bytes(self):
        files_before = {p.name: p.read_bytes() for p in self.root.glob("*.png")}
        code, report = self.run_contract("--output", self.root / "report.json")
        self.assertEqual(code, 0)
        self.assertEqual(report["edits"][0]["protected_pixels"], 1)
        self.assertEqual(report["edits"][0]["allowed_pixels"], 1)
        self.assertEqual(report["edits"][0]["changed_pixels"], 1)
        self.assertEqual(report["edits"][0]["outside_changed_pixels"], 0)
        self.assertEqual(json.loads((self.root / "report.json").read_text()), report)
        self.assertEqual(files_before, {p.name: p.read_bytes() for p in self.root.glob("*.png")})

    def test_hidden_RGB_change_outside_mask_fails(self):
        self.save("candidate.png", [(70, 80, 90, 255), (41, 50, 60, 0)])
        self.contract["edits"][0]["candidate_sha256"] = self.sha("candidate.png")
        code, report = self.run_contract()
        self.assertEqual(code, 1)
        self.assertEqual(report["edits"][0]["outside_changed_pixels"], 1)

    def test_declared_unchanged_file_SHA_failure(self):
        self.contract = {"unchanged": [{"path": "baseline.png", "sha256": "0" * 64}]}
        code, report = self.run_contract()
        self.assertEqual(code, 1)
        self.assertFalse(report["unchanged"][0]["sha256_matches"])

    def test_dimensions_cannot_hide_an_outside_pixel(self):
        self.save("candidate.png", [(70, 80, 90, 255)], size=(1, 1))
        self.contract["edits"][0]["candidate_sha256"] = self.sha("candidate.png")
        code, report = self.run_contract()
        self.assertEqual(code, 1)
        self.assertIn("dimensions_mismatch", [x["code"] for x in report["errors"]])

    def test_nonbinary_mask_fails(self):
        self.save("mask.png", [(255, 255, 255, 255), (1, 0, 0, 255)])
        self.contract["edits"][0]["allowed_mask_sha256"] = self.sha("mask.png")
        code, report = self.run_contract()
        self.assertEqual(code, 1)
        self.assertIn("mask_not_binary_red", [x["code"] for x in report["errors"]])

    def test_all_white_mask_cannot_certify_local_protection(self):
        self.save("mask.png", [(255, 255, 255, 255)] * 2)
        self.contract["edits"][0]["allowed_mask_sha256"] = self.sha("mask.png")
        code, report = self.run_contract()
        self.assertEqual(code, 1)
        self.assertIn("no_protected_pixels", [x["code"] for x in report["errors"]])

    def test_missing_or_malformed_contract_fields_fail(self):
        invalids = [None, {}, {"edits": "wrong"}, {"unchanged": [{}]},
                    {"unchanged": [{"path": "baseline.png", "sha256": "bad"}]},
                    {"edits": [dict(self.contract["edits"][0], mask_origin="")]}]
        missing = dict(self.contract["edits"][0])
        missing.pop("baseline_sha256")
        invalids.append({"edits": [missing]})
        for value in invalids:
            with self.subTest(contract=value):
                self.contract = value
                code, _ = self.run_contract()
                self.assertEqual(code, 2)

    def test_missing_input_is_IO_error(self):
        self.contract["edits"][0]["baseline"] = "missing.png"
        code, _ = self.run_contract()
        self.assertEqual(code, 2)

    def test_report_cannot_overwrite_an_image_or_contract(self):
        for target in [self.root / "baseline.png", self.contract_path]:
            with self.subTest(target=target):
                original = (self.root / "baseline.png").read_bytes()
                code, _ = self.run_contract("--output", target)
                self.assertEqual(code, 2)
                self.assertEqual((self.root / "baseline.png").read_bytes(), original)
                self.assertEqual(json.loads(self.contract_path.read_text()), self.contract)

    def test_black_mask_checks_entire_original_RGBA(self):
        (self.root / "candidate.png").write_bytes((self.root / "baseline.png").read_bytes())
        self.contract["edits"][0]["candidate_sha256"] = self.sha("candidate.png")
        self.save("mask.png", [0, 0], mode="L")
        self.contract["edits"][0]["allowed_mask_sha256"] = self.sha("mask.png")
        code, report = self.run_contract()
        self.assertEqual(code, 0)
        self.assertEqual(report["edits"][0]["protected_pixels"], 2)
        self.assertEqual(report["edits"][0]["allowed_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
