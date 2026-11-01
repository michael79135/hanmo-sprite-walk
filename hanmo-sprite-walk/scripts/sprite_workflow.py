#!/usr/bin/env python3
"""Offline reference -> direction masters -> measured sprite sequence workflow.

This tool handles six-active-frame walk/run with stand-first seven-slot delivery,
and historical six-slot sources. Other frame counts/actions use a separate explicit
workflow; never duplicate/pad frames to fit this helper.
This tool NEVER generates artwork, calls a model, removes a background, poses
limbs, or certifies visual quality. Python 3.10+ and Pillow are required.
All accepted originals and derived PNGs are immutable, hash-checked candidates.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from PIL import Image, ImageDraw

DIRECTIONS = ("right", "left", "down", "up")
ACTIONS = ("walk", "run")
SIZE = 96
FRAMES = 6
SCHEMA = "hanmo-sprite-workflow-v1"
SEQUENCE_MODES = ("separate_stand", "legacy_embedded_stand")
PRESENTATION_LAYOUTS = ("stand_first7", "sequence_frames")


class WorkflowError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise WorkflowError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha(im):
    return hashlib.sha256(im.convert("RGBA").tobytes()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"Cannot read JSON {path}: {exc}") from exc


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + os.urandom(6).hex())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def job_path(value):
    path = Path(value).expanduser().resolve()
    return path.parent if path.name == "manifest.json" else path


def load_job(value):
    root = job_path(value)
    data = read_json(root / "manifest.json")
    need(data.get("schema") == SCHEMA, "Unsupported workflow manifest")
    sequence_mode(data)
    presentation_layout(data)
    for relative, expected in data.get("protected_files", {}).items():
        path = root / relative
        need(path.is_file() and sha(path) == expected,
             f"Immutable artifact missing or changed: {relative}; restore it or start a new job")
    return root, data


def sequence_mode(data):
    # Missing field is the historical six-slot contract, never silently migrate it.
    mode = data.get("sequence_mode", "legacy_embedded_stand")
    need(mode in SEQUENCE_MODES, f"Unsupported sequence_mode: {mode}")
    return mode


def presentation_layout(data):
    # Old manifests keep their original numbered sequence, even if stand is separate.
    layout = data.get("presentation_layout", "sequence_frames")
    need(layout in PRESENTATION_LAYOUTS, f"Unsupported presentation_layout: {layout}")
    need(layout != "stand_first7" or sequence_mode(data) == "separate_stand",
         "stand_first7 requires separate_stand with all six active frames")
    return layout


def protect(root, data, paths):
    for path in paths:
        relative = str(Path(path).relative_to(root))
        digest = sha(path)
        need(relative not in data["protected_files"], f"Refusing to replace protected file: {relative}")
        data["protected_files"][relative] = digest


def review_states():
    return {"structure": "pending", "independent_evidence": "pending",
            "visual": "needs_review", "user": "not_reviewed", "runtime": "not_requested"}


def refresh_state(data):
    data["ingested_tasks"] = sum(t.get("active_attempt") is not None for t in data["tasks"])
    data["delivery_status"] = ("all_tasks_ingested_candidates" if data["ingested_tasks"] == len(data["tasks"])
                               else "partial")
    data["updated_at"] = utcnow()


def save_job(root, data):
    refresh_state(data)
    write_json(root / "manifest.json", data)


def input_image(path, require_size=False):
    path = Path(path).expanduser().resolve()
    need(path.is_file(), f"Missing source: {path}")
    try:
        with Image.open(path) as opened:
            need(getattr(opened, "n_frames", 1) == 1, "Animated inputs must be supplied as an explicit still PNG")
            im = opened.convert("RGBA")
            if require_size:
                need(opened.format == "PNG" and opened.mode == "RGBA" and opened.size == (SIZE, SIZE),
                     "Direction master must already be a 96x96 RGBA PNG. If the model returned a larger canvas, first map its complete square canvas to 96px with one declared fixed mapping, never per-frame bbox resize/centering")
            extrema = im.getchannel("A").getextrema()
            need(extrema[0] == 0 and extrema[1] > 0,
                 "Source must contain transparent background and visible artwork. Opaque backgrounds are rejected; "
                 "supply an explicitly transparency-processed source, not automatic white removal")
            return path, im.copy()
    except OSError as exc:
        raise WorkflowError(f"Cannot decode source {path}: {exc}") from exc


def prompt_for(data, direction, action):
    separate = sequence_mode(data) == "separate_stand"
    cells = ("All six cells are ACTIVE movement poses covering both alternating half-cycles. "
             "The natural closed-foot master is a separate STAND / START asset; do not place it in the repeating cycle. "
             "Preserve the first active contact pose as well as the other five poses. "
             "If an existing engine requires WALK1/RUN1 to equal STAND, export a separate seven-slot wrapper "
             "[STAND, active1, active2, active3, active4, active5, active6] and loop slots 2-7."
             if separate else
             "Historical embedded-stand format: the first cell is a supplied natural closed-foot master placeholder. "
             "The pipeline copies that master byte-for-byte into STAND / WALK1 / RUN1; only cells 2-6 provide movement. "
             "Do not use this five-active-frame compatibility format for a new complete six-active-frame cycle.")
    gait = ("WALK only: keep at least one supporting foot on the ground; include low passing-foot poses, "
            "alternating contact/support and opposite arm swing. Do not add a flight phase."
            if action == "walk" else
            "RUN only: draw a distinct longer stride, bent-knee recovery and a brief airborne phase, "
            "with coordinated opposite arm swing. Running must not be a sped-up walking sequence.")
    return f"""# {data['name']} | {direction.upper()} {action.upper()} | source_required / provider_pending
Route: builtin_imagegen. This file is an instruction, not a generated result.
Attachments required: original reference archive + observed natural {direction} master PNG.
First inspect reference identity and direction; prepare and record a separate natural {direction} master before this sequence.
Never mirror another direction. Never replace a character's face with a shared template or use a common skeleton.

Delivery target: one source sequence containing ONLY this character's {direction} {action}.
This target does not require one-shot generation. Before using a motion template, inspect its complete two-half gait and passing/contact poses; historical approval is not proof it meets the current limb-motion rules.
If its legs are ambiguous or it repeats stand/small-stride/large-stride without clear passing, use it only for pixel style. Establish and inspect opposite-support key poses from the same identity master before generating transitions.
The attached 96x96 master is the square input canvas. Do not force the overall output sheet to be square; actual output layout is measured.
Six cells in reading order; 3 columns x 2 rows is a proposed layout, not a crop assumption.
Use a transparent RGBA background with clear transparent gaps. Keep full silhouettes, all clothing, hands and feet inside each cell.
Use a single fixed square cell canvas, scale and ground anchor across all six cells. Preserve original pixel technique and character identity.
{cells}
The active poses must genuinely alternate anatomical arms and legs, with intermediate passing poses and connected head/neck/shoulders.
{gait}
Hands empty unless the user's current brief explicitly overrides this. Preserve deliberate identity-specific proportions.
Never fake limb movement by whole-body translation/bounce, separate bbox scaling, or fixed-head pasting.
Use the SAME direction master plus ORIGINAL reference as identity inputs for this whole sequence; do not chain candidate frames as new identity references.

After generation archive the actual source and inspect its real spacing and all six full silhouettes.
Measure common cell_size, origin, column_pitch, row_pitch and cols from the returned image; include concrete observation evidence.
Do not assume a requested 3x2 layout was obeyed. No per-frame bbox centering/scaling/foot snapping.
Independent limb/phase observation and browser review remain required; successful generation or crops do not prove quality.
"""


def init_job(args):
    layout = args.presentation_layout or ("stand_first7" if args.sequence_mode == "separate_stand" else "sequence_frames")
    presentation_layout({"sequence_mode": args.sequence_mode, "presentation_layout": layout})
    reference = Path(args.reference).expanduser().resolve()
    need(reference.is_file(), f"Missing reference: {reference}")
    try:
        with Image.open(reference) as im:
            im.verify()
    except OSError as exc:
        raise WorkflowError(f"Reference is not a readable image: {exc}") from exc
    motion = Path(args.motion_reference).expanduser().resolve() if args.motion_reference else None
    if motion:
        need(motion.is_file(), f"Missing motion reference: {motion}")
        try:
            with Image.open(motion) as im:
                im.verify()
        except OSError as exc:
            raise WorkflowError(f"Unreadable motion reference: {exc}") from exc
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = args.name.strip()
    need(bool(name), "Name cannot be empty")
    slug = re.sub(r"[^\w.-]+", "-", name, flags=re.UNICODE).strip(".-") or "character"
    n = 1
    while True:
        root = output / f"{slug}-v{n:03d}"
        try:
            root.mkdir()
            break
        except FileExistsError:
            n += 1
    try:
        original = root / "reference" / ("original" + reference.suffix.lower())
        original.parent.mkdir()
        shutil.copyfile(reference, original)
        data = {"schema": SCHEMA, "name": name, "created_at": utcnow(),
                "requested_route": "builtin_imagegen", "network_calls": False,
                "canvas_size": [SIZE, SIZE], "frame_count": FRAMES,
                "sequence_mode": args.sequence_mode,
                "presentation_layout": layout,
                "actions": args.actions, "directions": args.directions,
                "reference": {"path": str(original.relative_to(root)), "sha256": sha(original),
                              "original_path": str(reference), "observed_direction": None,
                              "direction_observation_evidence": None},
                "masters": {d: {"status": "source_required", "provider_status": "provider_pending",
                                 "path": None, "evidence": None, "visual": "needs_review",
                                 "user": "not_reviewed", "natural_closed_feet_verified": False}
                            for d in args.directions},
                "tasks": [], "review": review_states(), "protected_files": {},
                "deliveries": [], "artwork_modified_by_script": False,
                "per_frame_bbox_normalization": False,
                "runtime_integration": "not_requested"}
        paths = [original]
        data["motion_reference"] = None
        if motion:
            motion_copy = root / "reference" / ("motion" + motion.suffix.lower())
            shutil.copyfile(motion, motion_copy)
            paths.append(motion_copy)
            data["motion_reference"] = {"path": str(motion_copy.relative_to(root)), "sha256": sha(motion_copy),
                                        "role": "motion_only_not_character_identity", "observed_direction": None}
        for action in args.actions:
            for direction in args.directions:
                task_id = f"{direction}-{action}"
                prompt = root / "prompts" / f"{task_id}.md"
                prompt.parent.mkdir(exist_ok=True)
                text = prompt_for(data, direction, action)
                if motion:
                    text += "\nOptional motion reference is archived. Use it ONLY for motion/pose timing, never character identity. Inspect its direction/action first. Uniform sampling of a GIF does not preserve its original hold durations and does not select six appropriate motion phases; independently choose and review poses/timing before use. No motion model/API is called by this tool.\n"
                prompt.write_text(text, encoding="utf-8")
                paths.append(prompt)
                data["tasks"].append({"id": task_id, "direction": direction, "action": action,
                                      "status": "source_required", "provider_status": "provider_pending",
                                      "prompt": str(prompt.relative_to(root)),
                                      "master_required": direction, "active_attempt": None, "attempts": [],
                                      "generation_provenance": "pending",
                                      "review": review_states()})
        protect(root, data, paths)
        save_job(root, data)
        return {"job": str(root), "manifest": str(root / "manifest.json"), "status": "partial",
                "next": "Observe reference direction, prepare each natural master, then set-master before imagegen sequences"}
    except Exception:
        shutil.rmtree(root)
        raise


def set_master(args):
    root, data = load_job(args.job)
    need(args.direction in data["directions"], "Direction is outside this job")
    need(bool(args.evidence.strip()), "Independent natural-stand observation evidence is required")
    need(args.closed_feet_verified, "Pass --closed-feet-verified only after independently observing natural closed feet and relaxed arms")
    source, im = input_image(args.source, require_size=True)
    master = data["masters"][args.direction]
    if master["path"]:
        need(sha(source) == master["sha256"], "Master is immutable. Start a new job/version to replace it")
        return {"job": str(root), "direction": args.direction, "status": "already_recorded"}
    path = root / "masters" / args.direction / "stand.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), "Refusing to overwrite existing master")
    shutil.copyfile(source, path)
    master.update({"status": "observed_candidate", "provider_status": "source_supplied",
                   "path": str(path.relative_to(root)), "sha256": sha(path),
                   "source_path": str(source), "evidence": args.evidence.strip(),
                   "natural_closed_feet_verified": True, "recorded_at": utcnow(),
                   "alpha_bbox": im.getchannel("A").getbbox()})
    if args.reference_direction:
        data["reference"]["observed_direction"] = args.direction
        data["reference"]["direction_observation_evidence"] = args.evidence.strip()
    protect(root, data, [path])
    save_job(root, data)
    return {"job": str(root), "direction": args.direction, "status": "observed_candidate",
            "user_approval": "not_inferred"}


def integer(value, label, minimum=None):
    need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
         and int(value) == value, f"{label} must be an integer measured in source pixels")
    value = int(value)
    need(minimum is None or value >= minimum, f"{label} must be >= {minimum}")
    return value


def mapping_spec(value, source_size):
    need(isinstance(value, dict), "Mapping must be an object")
    allowed = {"cell_size", "origin", "column_pitch", "row_pitch", "cols", "evidence", "loop_indices", "notes"}
    need(not (set(value) - allowed),
         "Unsupported mapping fields (per-frame mappings, bbox normalization, isolation/cleanup are not performed): "
         + ", ".join(sorted(set(value) - allowed)))
    for field in ("cell_size", "origin", "column_pitch", "row_pitch", "cols", "evidence"):
        need(field in value, f"Measured mapping is missing {field}")
    need(isinstance(value["evidence"], str) and bool(value["evidence"].strip()), "Mapping evidence cannot be empty")
    size, origin = value["cell_size"], value["origin"]
    need(isinstance(size, list) and len(size) == 2, "cell_size must be [width,height]")
    need(isinstance(origin, list) and len(origin) == 2, "origin must be [x,y]")
    w, h = [integer(v, "cell_size", 1) for v in size]
    need(w == h, "All source cells must share one square canvas")
    ox, oy = [integer(v, "origin", 0) for v in origin]
    dx = integer(value["column_pitch"], "column_pitch", 1)
    dy = integer(value["row_pitch"], "row_pitch", 1)
    cols = integer(value["cols"], "cols", 1)
    need(cols <= FRAMES, "cols cannot exceed the six frame slots")
    boxes = []
    for i in range(FRAMES):
        x, y = ox + (i % cols) * dx, oy + (i // cols) * dy
        box = [x, y, x + w, y + h]
        need(box[2] <= source_size[0] and box[3] <= source_size[1],
             f"Frame {i + 1} crop {box} is out of bounds for source {source_size}")
        boxes.append(box)
    loops = value.get("loop_indices")
    if loops is not None:
        need(isinstance(loops, list) and bool(loops), "loop_indices must be a nonempty 1-based array")
        loops = [integer(v, "loop_indices", 1) for v in loops]
        need(max(loops) <= FRAMES and len(set(loops)) == len(loops), "Invalid or repeated loop index")
    normalized = {"cell_size": [w, h], "origin": [ox, oy], "column_pitch": dx,
                  "row_pitch": dy, "cols": cols, "evidence": value["evidence"].strip(), "loop_indices": loops}
    if "notes" in value:
        normalized["notes"] = value["notes"]
    return normalized, boxes


def crop_warnings(source, boxes):
    warnings = []
    for i, box in enumerate(boxes):
        alpha = source.crop(box).getchannel("A")
        w, h = alpha.size
        edge_names = []
        for name, edge in (("left", (0, 0, 1, h)), ("top", (0, 0, w, 1)),
                           ("right", (w-1, 0, w, h)), ("bottom", (0, h-1, w, h))):
            if alpha.crop(edge).getbbox():
                edge_names.append(name)
        if edge_names:
            warnings.append({"code": "crop_edge_alpha", "frame": i+1, "edges": edge_names,
                             "meaning": "Visible pixels meet crop boundary; independently check clipping/cross-cell contamination"})
        for j, other in enumerate(boxes[:i]):
            overlap = (max(box[0], other[0]), max(box[1], other[1]),
                       min(box[2], other[2]), min(box[3], other[3]))
            if overlap[2] > overlap[0] and overlap[3] > overlap[1]:
                if source.crop(overlap).getchannel("A").getbbox():
                    warnings.append({"code": "shared_crop_alpha", "frames": [j+1, i+1],
                                     "overlap": list(overlap),
                                     "meaning": "Common crops overlap visible source pixels; redraw source or remeasure, never erase a neighbour"})
    return warnings


def contact_sheet(frames):
    out = Image.new("RGBA", (SIZE*3, SIZE*2), (116, 127, 123, 255))
    for i, frame in enumerate(frames):
        out.alpha_composite(frame, ((i % 3)*SIZE, (i//3)*SIZE))
    return out


def generation_record(path):
    if not path:
        return None
    record = read_json(path)
    need(isinstance(record, dict), "Generation record must be a JSON object")
    need(record.get("kind") in ("tool_generation", "fixture_replay"),
         "Generation record kind must be tool_generation or fixture_replay")
    for key in ("tool", "model", "prompt", "executed_at"):
        need(isinstance(record.get(key), str) and bool(record[key].strip()),
             f"Generation record requires nonempty {key}; use unknown_from_tool if the model is undisclosed")
    try:
        timestamp = datetime.fromisoformat(record["executed_at"].replace("Z", "+00:00"))
        need(timestamp.tzinfo is not None, "Generation executed_at must include a timezone")
    except ValueError as exc:
        raise WorkflowError("Generation executed_at must be ISO8601 with timezone") from exc
    attachments = record.get("attachments")
    need(isinstance(attachments, list) and bool(attachments), "Generation record requires attachment paths and roles")
    for attachment in attachments:
        need(isinstance(attachment, dict) and all(isinstance(attachment.get(k), str) and attachment[k].strip()
             for k in ("path", "role")), "Each attachment requires path and its identity/master/motion purpose")
    if record["kind"] == "tool_generation":
        need(record.get("provider_kind") in ("builtin", "external"), "Generation record requires provider_kind builtin/external")
        if record["provider_kind"] == "external":
            need(isinstance(record.get("external_task_id"), str) and bool(record["external_task_id"].strip()),
                 "External generation record requires external_task_id")
    return record


def ingest(args):
    root, data = load_job(args.job)
    task = next((t for t in data["tasks"] if t["id"] == args.task), None)
    need(task is not None, f"Unknown task: {args.task}")
    master = data["masters"][task["direction"]]
    need(master["path"] and master["natural_closed_feet_verified"], "Record an independently observed direction master with set-master first")
    source_path, source = input_image(args.source)
    record = generation_record(args.generation_record)
    provenance = "pending" if record is None else "fixture_replay" if record["kind"] == "fixture_replay" else "recorded_claim"
    raw_mapping = read_json(args.mapping)
    mapping, boxes = mapping_spec(raw_mapping, source.size)
    separate = sequence_mode(data) == "separate_stand"
    if separate:
        complete_cycle = list(range(1, FRAMES + 1))
        need(mapping["loop_indices"] is None or mapping["loop_indices"] == complete_cycle,
             "separate_stand requires complete loop_indices [1,2,3,4,5,6]; "
             "do not drop or reorder active frames. If a new order is needed, reorganize the source sequence first")
        mapping["loop_indices"] = complete_cycle
    candidates = [source.crop(b).resize((SIZE, SIZE), Image.Resampling.NEAREST) for b in boxes]
    for i, candidate in enumerate(candidates):
        need(candidate.getchannel("A").getbbox() is not None, f"Source frame {i+1} is empty")
    master_path = root / master["path"]
    with Image.open(master_path) as opened:
        first = opened.convert("RGBA")
    frames = candidates if separate else [first.copy()] + candidates[1:]
    seen = {}
    duplicate_active = []
    master_digest = pixel_sha(first)
    warnings = crop_warnings(source, boxes)
    activity = enumerate(frames, start=1) if separate else enumerate(frames[1:], start=2)
    for i, frame in activity:
        digest = pixel_sha(frame)
        if digest in seen:
            duplicate_active.append([seen[digest], i])
        seen[digest] = i
        if digest == master_digest:
            warnings.append({"code": "master_repeated_in_activity", "frames": [i] if separate else [1, i],
                             "meaning": "Master equals an activity frame; retain source and inspect whether movement is missing"})
    accepted = not duplicate_active
    if duplicate_active:
        warnings.append({"code": "duplicate_active_frames", "pairs": duplicate_active,
                         "meaning": "Rejected as active attempt; original source and candidate frames retained for repair evidence"})
    base = root / "attempts" / task["id"]
    base.mkdir(parents=True, exist_ok=True)
    serial = len(task["attempts"]) + 1
    dest = base / f"attempt-{serial:03d}"
    need(not dest.exists(), f"Refusing to overwrite existing attempt {dest}")
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=base))
    try:
        archived = temporary / ("source" + source_path.suffix.lower())
        shutil.copyfile(source_path, archived)
        if record is not None:
            shutil.copyfile(Path(args.generation_record).expanduser().resolve(), temporary / "generation_record.json")
        write_json(temporary / "mapping.json", mapping)
        write_json(temporary / "mapping_submitted.json", raw_mapping)
        (temporary / "frames").mkdir()
        if not separate:
            candidates[0].save(temporary / "generated_slot1_unused.png")
        records = []
        for i, frame in enumerate(frames):
            path = temporary / "frames" / f"{i+1:02d}.png"
            if i == 0 and not separate:
                shutil.copyfile(master_path, path)
            else:
                frame.save(path)
            alpha = frame.getchannel("A")
            records.append({"frame": i+1, "path": f"{dest.relative_to(root)}/frames/{i+1:02d}.png",
                            "sha256": sha(path), "pixel_sha256": pixel_sha(frame),
                            "source_crop": boxes[i] if separate or i else None,
                            "generated_source_crop": boxes[i], "alpha_bbox": alpha.getbbox(),
                            "provenance": "exact_master_bytes" if i == 0 and not separate else "shared_canvas_nearest_mapping"})
        stand = temporary / "stand.png"
        shutil.copyfile(master_path, stand)
        shutil.copyfile(master_path, temporary / "start.png")
        atlas = Image.new("RGBA", (SIZE*FRAMES, SIZE))
        for i, frame in enumerate(frames):
            atlas.paste(frame, (SIZE*i, 0))
        atlas.save(temporary / "atlas.png")
        if separate:
            wrapper = Image.new("RGBA", (SIZE * (FRAMES + 1), SIZE))
            for i, frame in enumerate([first] + frames):
                wrapper.paste(frame, (SIZE * i, 0))
            wrapper.save(temporary / "legacy7_atlas.png")
        contact = contact_sheet(frames)
        contact.save(temporary / "contact_native.png")
        contact.resize((contact.width*3, contact.height*3), Image.Resampling.NEAREST).save(temporary / "contact_3x.png")
        evidence = {"task": task["id"], "source_sha256": sha(archived), "source_dimensions": list(source.size),
                    "mapping": mapping, "master_sha256": master["sha256"], "frames": records,
                    "sequence_mode": sequence_mode(data),
                    "stand": "stand.png", "start": "start.png",
                    "legacy7_wrapper": {"path": "legacy7_atlas.png", "loop_indices": list(range(2, 8)),
                                        "slot_roles": ["stand"] + [f"active{n}" for n in range(1, 7)]} if separate else None,
                    "generation_provenance": provenance, "generation_claims_verified_by_script": False,
                    "warnings": warnings, "review": review_states(),
                    "limb_review": {"anatomical_left_arm": None, "anatomical_right_arm": None,
                                    "anatomical_left_leg": None, "anatomical_right_leg": None},
                    "frame_observations": [{"frame": n, "identity": None, "neck_shoulders": None,
                                            "arms": None, "legs": None, "phase": None, "decision": None}
                                           for n in range(1, FRAMES+1)],
                    "pairwise_observations": [{"frames": [a, b], "observed": None}
                                              for a in range(1, FRAMES+1) for b in range(a+1, FRAMES+1)],
                    "artwork_modified": False, "per_frame_bbox_normalization": False,
                    "loop_range_review": ("complete_cycle_default_unreviewed" if separate and raw_mapping.get("loop_indices") is None
                                          else "observed_mapping_declared" if mapping["loop_indices"] else "pending")}
        evidence["review"]["structure"] = "rejected_duplicates" if not accepted else "needs_review" if warnings else "pass"
        write_json(temporary / "evidence.json", evidence)
        # Observer-owned file is deliberately mutable; this is evidence to fill,
        # never an automatic promotion of visual/user/runtime status.
        write_json(temporary / "review_observations.json", {
            "status": "pending", "limb_review": evidence["limb_review"],
            "frame_observations": evidence["frame_observations"],
            "pairwise_observations": evidence["pairwise_observations"],
            "continuous_preview_observation": None, "loop_transition_observation": None,
            "user_approval_evidence": None, "runtime_evidence": None})
        temporary.rename(dest)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    paths = sorted(p for p in dest.rglob("*") if p.is_file() and p.name != "review_observations.json")
    protect(root, data, paths)
    attempt = {"id": serial, "path": str(dest.relative_to(root)), "created_at": utcnow(),
               "source_sha256": sha(dest / archived.name), "master_sha256": master["sha256"],
               "mapping": mapping, "frames": records, "warnings": warnings,
               "review": evidence["review"], "accepted": accepted,
               "generation_provenance": provenance,
               "generation_record": str((dest / "generation_record.json").relative_to(root)) if record is not None else None,
               "observations": str((dest / "review_observations.json").relative_to(root))}
    attempt["source"] = str((dest / archived.name).relative_to(root))
    task["attempts"].append(attempt)
    if accepted:
        task["active_attempt"] = serial
        task["status"] = "ingested_candidate"
        task["provider_status"] = "source_supplied_unverified"
        task["generation_provenance"] = provenance
        task["review"] = dict(evidence["review"])
    else:
        task["provider_status"] = "returned_source_rejected"
        if task["active_attempt"] is None:
            task["status"] = "source_required"
    save_job(root, data)
    return {"job": str(root), "task": task["id"], "attempt": serial,
            "structure": evidence["review"]["structure"], "warnings": warnings,
            "accepted": accepted, "generation_provenance": provenance, "visual": "needs_review", "delivery_status": data["delivery_status"]}


PREVIEW = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>人物帧候选审阅</title><style>
body{font:15px system-ui;background:#20262c;color:#edf0f3;margin:24px;max-width:1100px}h1{font-size:22px}label{display:inline-block;margin:5px 12px 5px 0}button,input,select{font:inherit;padding:5px;background:#343f4b;color:white;border:1px solid #65717d;border-radius:4px}input[type=number]{width:66px}#loop{width:150px}.warn{color:#ffcf89}.failed{color:#fff1f0;background:#702a2e;border:2px solid #ff8686;padding:12px;font-weight:700}canvas{image-rendering:pixelated;image-rendering:crisp-edges}#stageViewport{width:100%;height:330px;box-sizing:border-box;overflow:hidden;border:1px solid #839294}#stage{background:repeating-conic-gradient(#4b5759 0 25%,#525f61 0 50%) 0/24px 24px;display:block}#frames{display:flex;flex-wrap:wrap;gap:9px}figure{margin:4px;text-align:center}figure img{image-rendering:pixelated;image-rendering:crisp-edges;background:#65716e}p{line-height:1.5}.box{border:1px solid #596470;padding:12px;margin:12px 0}#frameCanvas{background:#65716e}</style>
<h1 id="title"></h1><p id="status" class="warn"></p><p>站姿与活动循环按任务格式分别展示；新任务保留六张活动帧，站姿不插入每圈循环。图像来自原PNG，未添加程序摆肢或上下弹跳。机械结构、独立观察、连续播放、用户认可与运行接入分别记录。</p>
<div class="box"><label>动作 <select id="action"></select></label><label>方向 <select id="direction"></select></label><button id="play">播放</button><button id="step">逐帧 +1</button><button id="stand">站姿/起步重置</button><label>FPS <input id="fps" type="number" value="8" min="1" max="30"></label><label>倍率 <select id="scale"><option>1</option><option>2</option><option selected>3</option><option>4</option></select></label><label>移动 <select id="mode"><option value="inplace">原地</option><option value="world">真实世界平移</option></select></label><label>每圈步幅 px <input id="stride" type="number" value="48" min="1" max="300"></label><details><summary>观察实验（高级）</summary><p>新任务按完整六张活动帧交付。此处临时筛选或调序只用于定位问题，不改变交付循环或认可状态。历史格式保留其声明范围。</p><label>循环模式 <select id="loopMode"><option value="full">全部循环帧</option><option value="custom">临时观察实验</option></select></label><label>帧序号 <input id="loop" value="1,2,3,4,5,6"></label><button id="apply">应用观察实验</button><p id="loopReadout"></p><p>新任务提供的七槽兼容图集的第 1 槽为站姿，第 2–7 槽为完整六帧活动循环；新七槽主图集以站姿为第1格，另存六活动图集；旧布局保持原顺序。</p></details><p id="info"></p><p id="taskStatus" class="warn"></p><p class="warn" id="phase"></p><div id="stageViewport"><canvas id="stage" width="1000" height="330"></canvas></div></div>
<p>原尺寸逐帧联系图（固定画布；可见接边与跨格问题须退回源图或重测，不能自动删边）</p><div id="frames"></div><p id="warnings" class="warn"></p>
<script>const DATA=__DATA__;let loaded=[],standImage=null,current=null,playing=false,frame=0,loop=[0,1,2,3,4,5],last=0,elapsed=0,worldX=0,worldY=0,token=0;const $=id=>document.getElementById(id),canvas=$('stage'),ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;$('title').textContent=DATA.name+' · 人物帧候选';$('status').textContent=DATA.status+' | 已收编 '+DATA.ready+'/'+DATA.total+' 组；视觉 '+DATA.visual+'；用户未认可；运行未接入'+(DATA.visual==='needs_changes'?'；存在已退回动作，需修复后复审':'');$('status').className=DATA.visual==='needs_changes'?'failed':'warn';
function options(el,values){el.replaceChildren(...values.map(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;return o}))}options($('action'),DATA.actions);options($('direction'),DATA.directions);
function separate(){return current&&current.sequence_mode==='separate_stand'}
function standFirst(){return current&&current.presentation_layout==='stand_first7'}
function motionName(){return current&&current.action==='run'?'跑步':'走路'}
function displaySlot(index){return standFirst()?index+2:index+1}
function setLoopInput(){$('loop').value=loop.map(displaySlot).join(',')}
function reset(){playing=false;$('play').textContent='播放';frame=separate()?-1:0;worldX=worldY=0;elapsed=0;last=0;draw()}
function updateInfo(){let speed=(+$('stride').value||48)*(+$('fps').value||8)/loop.length;let shown=standFirst()?(frame<0?'第1张 / 7 · 自然站立':'第'+(frame+2)+'张 / 7 · '+motionName()+(frame+1)+'/6'):frame<0?'STAND / START（独立站姿）':'F'+(frame+1)+' / 6';$('info').textContent=(current?current.task:'未收编')+' · '+shown+' · 速度 '+speed.toFixed(1)+'px/s';$('loopReadout').textContent='当前观察顺序（成品编号） ['+loop.map(displaySlot).join(',')+']';let failed=current&&current.visual==='needs_changes';$('taskStatus').textContent=current?(current.task+' · 视觉 '+current.visual+(failed?' · 独立观察已退回，需修复后复审；用户未认可':' · 候选待审；用户未认可')):'当前任务尚未收编';$('taskStatus').className=failed?'failed':'warn'}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);let scale=+$('scale').value,im=frame<0?standImage:loaded[frame];if(im){ctx.imageSmoothingEnabled=false;ctx.drawImage(im,Math.round(canvas.width/2-48*scale+worldX*scale),Math.round(canvas.height/2-48*scale+worldY*scale),96*scale,96*scale)}updateInfo()}
async function select(){let my=++token;current=DATA.tasks.find(t=>t.action===$('action').value&&t.direction===$('direction').value)||null;loaded=[];standImage=null;reset();$('frames').replaceChildren();$('warnings').textContent='';if(!current){$('phase').textContent='该任务 source_required / provider_pending，未生成或未收编。';draw();return}loop=(current.loop_indices||[1,2,3,4,5,6]).map(v=>v-1);setLoopInput();$('loopMode').value=separate()?'full':current.loop_indices?'custom':'full';$('phase').textContent=standFirst()?'第1张自然站立；第2–7张完整'+motionName()+'循环，站姿不混入每圈。请观察腰胯连接、换腿、摆臂与末帧回环。':separate()?'六张活动帧独立循环，站姿单独展示；请观察换腿、摆臂与末帧→首帧是否自然。':current.loop_indices?'来自mapping的显式候选循环，仍须独立观察与末帧→首帧检查。':'循环范围未核验。当前展示含起步的全部1–6帧；这不是已确认的活动循环。请观察后选择自定活动循环并填写范围。';let sources=separate()?[...current.images,current.stand_image]:current.images;let images=await Promise.all(sources.map(src=>new Promise((resolve,reject)=>{let im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src}))).catch(()=>null);if(my!==token)return;if(!images){$('phase').textContent='图像加载失败';return}if(separate())standImage=images.pop();loaded=images;let thumbnails=standFirst()?[standImage,...images]:images;thumbnails.forEach((im,i)=>{let f=document.createElement('figure');let cp=im.cloneNode();cp.width=cp.height=96;f.append(cp);let cap=document.createElement('figcaption');cap.textContent=standFirst()?(i===0?'第1张 · 自然站立':'第'+(i+1)+'张 · '+motionName()+i):'F'+(i+1)+(i===0&&!separate()?' 母图原字节':' 活动');f.append(cap);f.onclick=()=>{playing=false;$('play').textContent='播放';elapsed=last=0;frame=standFirst()?i-1:i;draw()};$('frames').append(f)});$('warnings').textContent=current.warnings.map(w=>w.code+' 成品编号 '+(w.frame?displaySlot(w.frame-1):w.frames?w.frames.map(n=>displaySlot(n-1)).join('/'):JSON.stringify((w.pairs||[]).map(pair=>pair.map(n=>displaySlot(n-1)))))).join('；');draw()}
function resizeStage(){let viewport=$('stageViewport'),width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight);if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;canvas.style.width=width+'px';canvas.style.height=height+'px';draw()}
function advance(){if(!loaded.length)return;let position=loop.indexOf(frame);frame=loop[position<0?0:(position+1)%loop.length];draw()}
$('play').onclick=()=>{if(!loaded.length)return;if(!playing&&frame<0)advance();playing=!playing;$('play').textContent=playing?'暂停':'播放';last=0};$('step').onclick=()=>{playing=false;$('play').textContent='播放';advance()};$('stand').onclick=reset;
$('apply').onclick=()=>{let values=$('loop').value.split(',').map(v=>Number(v.trim()));let first=standFirst()?2:1,lastSlot=standFirst()?7:6;if(!values.length||values.some(v=>!Number.isInteger(v)||v<first||v>lastSlot)||new Set(values).size!==values.length){$('phase').textContent='活动观察编号须为'+first+'–'+lastSlot+'的非重复整数列表，站姿不能混入循环';return}loop=values.map(v=>v-(standFirst()?2:1));$('loopMode').value='custom';reset();$('phase').textContent=separate()?'临时观察实验；交付仍须保留原序完整六帧，此操作不写入通过或认可状态。':'当前会话候选循环已更新，未写入认可状态。请检查全部肢体和末帧→首帧。';};
$('loopMode').onchange=()=>{if($('loopMode').value==='full'){loop=[0,1,2,3,4,5];setLoopInput();reset();$('phase').textContent=standFirst()?'完整'+motionName()+'循环为成品第2–7张；第1张自然站姿单独展示，真实自然度仍须观察。':separate()?'完整六张活动帧循环；站姿单独展示，真实自然度仍须观察。':'含起步全帧检查模式；活动循环仍未核验。'}else{$('phase').textContent=separate()?'临时观察实验用于定位问题；只有原序完整六帧循环用于本任务交付。':'请输入实测活动循环的帧序号并点应用；不会自动隐藏任何帧。'}};for(let id of ['action','direction'])$(id).onchange=select;for(let id of ['scale','mode','stride','fps'])$(id).onchange=()=>{if(id==='mode'){worldX=worldY=0}draw()};
function tick(now){if(playing&&loaded.length){let dt=last?Math.min((now-last)/1000,.1):0;elapsed+=dt;let fps=Math.max(1,Math.min(30,+$('fps').value||8));while(elapsed>=1/fps){elapsed-=1/fps;advance()}if($('mode').value==='world'){let speed=Math.max(1,+$('stride').value||48)*fps/loop.length;let d=$('direction').value;worldX+=(d==='right'?1:d==='left'?-1:0)*speed*dt;worldY+=(d==='down'?1:d==='up'?-1:0)*speed*dt;let scale=+$('scale').value;if(Math.abs(worldX)>canvas.width/(2*scale)+48)worldX=-worldX;if(Math.abs(worldY)>canvas.height/(2*scale)+48)worldY=-worldY}draw()}last=now;requestAnimationFrame(tick)}if(typeof ResizeObserver!=='undefined'){const observer=new ResizeObserver(resizeStage);observer.observe($('stageViewport'))}window.addEventListener('resize',resizeStage);resizeStage();select();requestAnimationFrame(tick);
</script></html>'''


def finalize(args):
    root, data = load_job(args.job)
    refresh_state(data)
    # Only a negative observation on the currently active attempt can lower
    # visual status. Positive claims never imply visual/user acceptance.
    for task in data["tasks"]:
        visual = "needs_review"
        if task["active_attempt"] is not None:
            active = task["attempts"][task["active_attempt"]-1]
            observation_path = root / active["observations"]
            observation = read_json(observation_path) if observation_path.is_file() else None
            if isinstance(observation, dict) and observation.get("status") == "needs_changes":
                visual = "needs_changes"
            active["review"]["visual"] = visual
        task["review"]["visual"] = visual
    data["review"]["visual"] = ("needs_changes" if any(t["review"]["visual"] == "needs_changes" for t in data["tasks"])
                                else "needs_review")
    serial = len(data["deliveries"]) + 1
    dest = root / "deliveries" / f"delivery-{serial:03d}"
    need(not dest.exists(), "Refusing to overwrite existing delivery")
    dest.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=dest.parent))
    try:
        layout = presentation_layout(data)
        stand_first = layout == "stand_first7"
        payload = {"name": data["name"], "status": data["delivery_status"],
                   "ready": data["ingested_tasks"], "total": len(data["tasks"]),
                   "actions": data["actions"], "directions": data["directions"], "tasks": [],
                   "visual": data["review"]["visual"], "sequence_mode": sequence_mode(data),
                   "presentation_layout": layout}
        separate = sequence_mode(data) == "separate_stand"
        atlas_records, presentations = [], []
        for action in data["actions"]:
            active_atlas = Image.new("RGBA", (SIZE*FRAMES, SIZE*len(data["directions"])))
            wrapper = Image.new("RGBA", (SIZE*(FRAMES+1), SIZE*len(data["directions"]))) if separate else None
            rows = []
            for row, direction in enumerate(data["directions"]):
                task = next(t for t in data["tasks"] if t["action"] == action and t["direction"] == direction)
                rows.append({"row": row, "direction": direction, "status": task["status"]})
                if task["active_attempt"] is None:
                    continue
                attempt = task["attempts"][task["active_attempt"]-1]
                stand_path = root / data["masters"][direction]["path"]
                stand_data = "data:image/png;base64," + base64.b64encode(stand_path.read_bytes()).decode("ascii")
                if wrapper is not None:
                    with Image.open(stand_path) as opened:
                        wrapper.paste(opened.convert("RGBA"), (0, SIZE*row))
                images = []
                for i, rec in enumerate(attempt["frames"]):
                    source_path = root / rec["path"]
                    images.append("data:image/png;base64," + base64.b64encode(source_path.read_bytes()).decode("ascii"))
                    with Image.open(source_path) as opened:
                        active_atlas.paste(opened.convert("RGBA"), (SIZE*i, SIZE*row))
                        if wrapper is not None:
                            wrapper.paste(opened.convert("RGBA"), (SIZE*(i+1), SIZE*row))
                # Delivery numbering is separate from six-active storage. Copies preserve PNG bytes.
                ordered_sources = ([stand_path] if stand_first else []) + [root / rec["path"] for rec in attempt["frames"]]
                slots = []
                for i, source_path in enumerate(ordered_sources):
                    slot = i + 1
                    is_stand = (stand_first or not separate) and i == 0
                    activity_frame = None if is_stand else slot - 1 if stand_first else slot
                    relative = Path("frames") / task["id"] / f"{slot:02d}.png"
                    path = temporary / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_path, path)
                    slots.append({"slot": slot, "role": "stand" if is_stand else "active",
                                  "active_frame": activity_frame, "path": str(relative),
                                  "source_path": str(source_path.relative_to(root)), "sha256": sha(path)})
                display_loop = ([v + 1 for v in attempt["mapping"]["loop_indices"]] if stand_first
                                else attempt["mapping"]["loop_indices"])
                presentation = {"task": task["id"], "action": action, "direction": direction,
                                "layout": layout, "slot_count": len(slots), "slots": slots,
                                "loop_indices": display_loop,
                                "activity_loop_indices": attempt["mapping"]["loop_indices"],
                                "loop_index_domain": "delivery_slots", "user_approved": False}
                presentations.append(presentation)
                payload["tasks"].append({"task": task["id"], "action": action, "direction": direction,
                                         "images": images, "loop_indices": attempt["mapping"]["loop_indices"],
                                         "stand_image": stand_data, "sequence_mode": sequence_mode(data),
                                         "presentation_layout": layout, "presentation": presentation,
                                         "warnings": attempt["warnings"], "visual": task["review"]["visual"]})
            primary = wrapper if stand_first else active_atlas
            atlas_path = temporary / f"{action}_atlas.png"
            primary.save(atlas_path)
            slot_roles = (["stand"] + [f"active{n}" for n in range(1, 7)] if stand_first else
                          [f"active{n}" for n in range(1, 7)] if separate else
                          ["stand"] + [f"active{n}" for n in range(2, 7)])
            record = {"action": action, "path": atlas_path.name, "sha256": sha(atlas_path),
                      "rows": rows, "missing_rows_are_transparent": True,
                      "sequence_mode": sequence_mode(data), "presentation_layout": layout,
                      "slot_roles": slot_roles, "loop_indices": list(range(2, 8)) if stand_first else
                      list(range(1, 7)) if separate else None}
            contact = Image.new("RGBA", primary.size, (116, 127, 123, 255))
            contact.alpha_composite(primary)
            contact_path = temporary / f"{action}_contact_native.png"
            contact.save(contact_path)
            contact.resize((contact.width*3, contact.height*3), Image.Resampling.NEAREST).save(temporary / f"{action}_contact_3x.png")
            record["contact_native"] = {"path": contact_path.name, "sha256": sha(contact_path),
                                        "same_slot_order_as_primary_atlas": True}
            if separate:
                active_path = temporary / f"{action}_active_atlas.png" if stand_first else atlas_path
                if stand_first:
                    active_atlas.save(active_path)
                record["active_atlas"] = {"path": active_path.name, "sha256": sha(active_path),
                                           "slot_roles": [f"active{n}" for n in range(1, 7)],
                                           "loop_indices": list(range(1, 7)), "loop_index_domain": "active_frames"}
                wrapper_path = temporary / f"{action}_legacy7_atlas.png"
                wrapper.save(wrapper_path)
                record["legacy7_wrapper"] = {
                    "path": wrapper_path.name, "sha256": sha(wrapper_path),
                    "slot_roles": ["stand"] + [f"active{n}" for n in range(1, 7)],
                    "loop_indices": list(range(2, 8)), "loop_index_domain": "delivery_slots"}
            atlas_records.append(record)
        # Escape '<' so names/evidence cannot terminate the script element.
        encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
        (temporary / "preview.html").write_text(PREVIEW.replace("__DATA__", encoded), encoding="utf-8")
        snapshot = json.loads(json.dumps(data))
        snapshot.pop("protected_files", None)
        # Snapshot caller-authored observations/provenance as data, without
        # treating their presence as visual acceptance or actual model proof.
        for task in snapshot["tasks"]:
            for attempt in task["attempts"]:
                observation = root / attempt["observations"]
                attempt["independent_observations"] = read_json(observation) if observation.is_file() else None
                attempt["independent_observations_sha256"] = sha(observation) if observation.is_file() else None
                record_path = attempt.get("generation_record")
                attempt["generation_record_content"] = read_json(root / record_path) if record_path else None
        snapshot["snapshot_generated_at"] = utcnow()
        snapshot["snapshot_atlases"] = atlas_records
        snapshot["snapshot_presentations"] = presentations
        snapshot["self_contained_preview"] = True
        snapshot["review"]["structure"] = ("pending" if not data["ingested_tasks"] else
            ("needs_review" if any(t["review"]["structure"] == "needs_review" for t in data["tasks"]) else "partial" if data["delivery_status"] == "partial" else "pass"))
        write_json(temporary / "manifest.json", snapshot)
        temporary.rename(dest)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    protect(root, data, sorted(p for p in dest.rglob("*") if p.is_file()))
    data["deliveries"].append({"id": serial, "path": str(dest.relative_to(root)),
                               "status": data["delivery_status"], "visual": data["review"]["visual"], "created_at": utcnow()})
    save_job(root, data)
    return {"job": str(root), "preview": str(dest / "preview.html"),
            "manifest": str(dest / "manifest.json"), "delivery_status": data["delivery_status"],
            "visual": data["review"]["visual"], "user": "not_reviewed", "runtime": "not_requested"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("init", help="Create a versioned offline plan; no model calls")
    q.add_argument("--reference", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--motion-reference", help="Optional image/GIF archive; movement only, never identity")
    q.add_argument("--name", required=True)
    q.add_argument("--actions", nargs="+", choices=ACTIONS, default=["walk"])
    q.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    q.add_argument("--sequence-mode", choices=SEQUENCE_MODES, default="separate_stand",
                   help="New jobs retain six active poses with separate STAND/START; legacy mode reserves slot 1 for stand")
    q.add_argument("--presentation-layout", choices=PRESENTATION_LAYOUTS,
                   help="Default: stand_first7 for six active frames, sequence_frames for legacy storage")
    q.set_defaults(func=init_job)
    q = sub.add_parser("set-master", help="Record independently observed natural closed-foot 96px master")
    q.add_argument("--job", required=True)
    q.add_argument("--direction", choices=DIRECTIONS, required=True)
    q.add_argument("--source", required=True)
    q.add_argument("--evidence", required=True)
    q.add_argument("--closed-feet-verified", action="store_true")
    q.add_argument("--reference-direction", action="store_true", help="This is the observed direction of the original reference")
    q.set_defaults(func=set_master)
    q = sub.add_parser("ingest", help="Archive transparent source and measured common grid; obey job's declared stand/active layout")
    q.add_argument("--job", required=True)
    q.add_argument("--task", required=True)
    q.add_argument("--source", required=True)
    q.add_argument("--generation-record", help="Optional actual tool/model/full-prompt/attachments/time JSON or fixture_replay; never include credentials")
    q.add_argument("--mapping", required=True, help="JSON path with cell_size/origin/pitches/cols/evidence; optional loop_indices")
    q.set_defaults(func=ingest)
    q = sub.add_parser("finalize", help="Write a new immutable delivery snapshot, including partial jobs")
    q.add_argument("--job", required=True)
    q.set_defaults(func=finalize)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            need(len(set(args.actions)) == len(args.actions), "Repeated actions are not allowed")
            need(len(set(args.directions)) == len(args.directions), "Repeated directions are not allowed")
        result = args.func(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("accepted") is False else 0
    except (WorkflowError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
