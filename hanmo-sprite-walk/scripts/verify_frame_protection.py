#!/usr/bin/env python3
"""Verify declared file/PNG protection without changing any source image.

Requires Pillow. Mask red=0 protects every original RGBA byte, red=255 permits
editing; mask alpha is ignored. Success proves only the declared protection,
not mask provenance, complete coverage, anatomy, visual quality or approval.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from PIL import Image


class ContractError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(record, key, label):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label}.{key}: expected a nonempty string")
    return value


def load_contract(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError("contract must be an object")
    groups = {key: data.get(key, []) for key in ("unchanged", "edits")}
    if any(not isinstance(value, list) for value in groups.values()):
        raise ContractError("unchanged and edits must be arrays")
    if not any(groups.values()):
        raise ContractError("contract must declare at least one file or edit")
    specifications = {
        "unchanged": [("path", "sha256")],
        "edits": [("baseline", "baseline_sha256"),
                  ("candidate", "candidate_sha256"),
                  ("allowed_mask", "allowed_mask_sha256")],
    }
    inputs = {path.resolve()}
    for group, records in groups.items():
        for index, record in enumerate(records):
            label = f"{group}[{index}]"
            if not isinstance(record, dict):
                raise ContractError(f"{label}: expected an object")
            for file_key, hash_key in specifications[group]:
                filename = _field(record, file_key, label)
                digest = _field(record, hash_key, label)
                if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    raise ContractError(f"{label}.{hash_key}: expected SHA-256 hex")
                record[hash_key] = digest.lower()
                location = Path(filename).expanduser()
                if not location.is_absolute():
                    location = path.parent / location
                record[file_key] = str(location.resolve())
                inputs.add(location.resolve())
            if group == "edits":
                _field(record, "mask_origin", label)
    return groups, inputs


def _image(path):
    with Image.open(path) as image:
        image.load()
        return image.format, image.copy()


def verify(groups):
    errors, unchanged, edits = [], [], []

    def check_hash(record, file_key, hash_key, label):
        actual = sha256(Path(record[file_key]))
        matched = actual == record[hash_key]
        if not matched:
            errors.append({"record": label, "code": "sha256_mismatch",
                           "field": file_key, "expected": record[hash_key],
                           "actual": actual})
        return matched

    for index, record in enumerate(groups["unchanged"]):
        matched = check_hash(record, "path", "sha256", f"unchanged[{index}]")
        unchanged.append({"path": record["path"], "sha256_matches": matched})

    for index, record in enumerate(groups["edits"]):
        label = f"edits[{index}]"
        error_count = len(errors)
        for file_key in ("baseline", "candidate", "allowed_mask"):
            check_hash(record, file_key, file_key + "_sha256", label)
        baseline_format, baseline = _image(record["baseline"])
        candidate_format, candidate = _image(record["candidate"])
        mask_format, mask = _image(record["allowed_mask"])
        result = {key: record[key] for key in
                  ("baseline", "candidate", "allowed_mask", "mask_origin")}
        edits.append(result)
        if (baseline_format != "PNG" or candidate_format != "PNG"
                or baseline.mode != "RGBA" or candidate.mode != "RGBA"):
            errors.append({"record": label, "code": "expected_RGBA_PNG"})
        elif baseline.size != candidate.size or baseline.size != mask.size:
            errors.append({"record": label, "code": "dimensions_mismatch",
                           "baseline": baseline.size, "candidate": candidate.size,
                           "mask": mask.size})
        elif mask_format != "PNG" or mask.mode not in ("L", "RGB", "RGBA"):
            errors.append({"record": label, "code": "expected_L_RGB_RGBA_PNG_mask"})
        else:
            red = mask.tobytes() if mask.mode == "L" else mask.getchannel("R").tobytes()
            if any(value not in (0, 255) for value in red):
                errors.append({"record": label, "code": "mask_not_binary_red"})
            else:
                original, current = baseline.tobytes(), candidate.tobytes()
                changed = outside = 0
                for pixel, allowed in enumerate(red):
                    start = pixel * 4
                    differs = original[start:start + 4] != current[start:start + 4]
                    changed += differs
                    outside += differs and allowed == 0
                allowed_count = red.count(255)
                protected_count = red.count(0)
                result.update(size=list(baseline.size), changed_pixels=changed,
                              allowed_pixels=allowed_count,
                              protected_pixels=protected_count,
                              outside_changed_pixels=outside)
                if protected_count == 0:
                    errors.append({"record": label, "code": "no_protected_pixels",
                                   "message": "Full redraw requires other acceptance checks; an all-white mask proves no local protection."})
                if outside:
                    errors.append({"record": label, "code": "protected_RGBA_changed",
                                   "outside_changed_pixels": outside})
        result["passed"] = len(errors) == error_count
    return {"passed": not errors, "unchanged": unchanged, "edits": edits,
            "errors": errors,
            "scope": "Only declared file hashes and original RGBA outside the supplied binary red masks.",
            "not_verified": ["mask origin truth", "coverage completeness", "anatomy",
                             "gait", "clarity", "user approval", "engine integration"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report; never an input path")
    args = parser.parse_args(argv)
    contract = args.contract.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else None
    try:
        groups, inputs = load_contract(contract)
        if output is not None and (output in inputs or any(
                output.exists() and source.exists() and output.samefile(source)
                for source in inputs)):
            raise ContractError("report output must not overwrite the contract or any input")
        report = verify(groups)
        code = 0 if report["passed"] else 1
    except (ContractError, OSError, ValueError) as error:
        report = {"passed": False, "errors": [{"code": "invalid_contract_or_IO",
                                                "message": str(error)}]}
        code = 2
        # Invalid input/output paths cannot be trusted as writable report targets.
        output = None
    report["exit_code"] = code
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is not None:
        try:
            output.write_text(rendered, encoding="utf-8")
        except OSError as error:
            print(json.dumps({"passed": False, "exit_code": 2,
                              "errors": [{"code": "report_IO", "message": str(error)}]}))
            return 2
    sys.stdout.write(rendered)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
