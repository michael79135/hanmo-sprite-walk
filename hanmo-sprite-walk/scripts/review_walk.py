#!/usr/bin/env python3
"""Read-only sprite QA. Requires Pillow. Never certifies natural motion."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops, ImageDraw


DIRECTIONS = ("right", "left", "down", "up")
CANVAS = (96, 96)
LIMBS = ("arm_a", "arm_b", "leg_a", "leg_b")


def blank_limb_observation(limb):
    """Human observations only; image similarity cannot fill joint positions."""
    joints = ("shoulder", "elbow", "wrist", "hand") if limb.startswith("arm") else (
        "hip", "knee", "ankle", "toe", "heel")
    return {"visibility": "UNREVIEWED",
            "native_joints_xy": dict.fromkeys(joints),
            "relative_to_torso_observation": "", "joint_angle_observation": "",
            "phase": "", "motion_direction": "", "observation": "",
            "keep_or_repair": "", "accepted": None}


def blank_limb_identity():
    return {"anatomical_side": "unknown", "near_or_far": "unknown",
            "screen_side_observation": "", "stable_visual_identifier": ""}


def blank_pair_review(direction, pair, frame_count, source_files):
    a, b = pair
    return {"direction": direction, "frames": pair, "source_files": source_files,
            "adjacent": b == a + 1, "loop_seam": a == 1 and b == frame_count,
            "half_cycle_pair": b - a == frame_count // 2,
            "comparison_basis": "same limb identity relative to observed torso; not global image displacement",
            "limbs": {limb: {"position_differs_relative_to_torso": None,
                              "pose_progression_observation": "",
                              "visibility_or_occlusion_evidence": "",
                              "decision": "UNREVIEWED"} for limb in LIMBS},
            "keep_or_repair": "", "accepted": None}


def rectangle(value):
    values = tuple(int(part) for part in value.split(","))
    if len(values) != 4 or not (0 <= values[0] < values[2] <= 96
                               and 0 <= values[1] < values[3] <= 96):
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1 within 96×96")
    return values


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_png(path, errors, size):
    try:
        with Image.open(path) as im:
            if im.format != "PNG" or im.mode != "RGBA" or im.size != size:
                errors.append(f"{path.name}: expected RGBA PNG {size}; got {im.format} {im.mode} {im.size}")
                return None
            return im.copy()
    except (OSError, ValueError) as error:
        errors.append(f"{path.name}: cannot decode PNG: {error}")
        return None


def threshold_alpha(im, threshold):
    return im.getchannel("A").point(lambda a: 255 if a > threshold else 0)


def centroid(mask):
    width, height = mask.size
    pixels = list(mask.tobytes())
    mass = sum(pixels)
    if not mass:
        return None
    return [round(sum((i % width) * p for i, p in enumerate(pixels)) / mass, 3),
            round(sum((i // width) * p for i, p in enumerate(pixels)) / mass, 3)]


def components(mask):
    """8-connected components; detached costume pieces are review hints."""
    width, height = mask.size
    occupied = {i for i, value in enumerate(mask.tobytes()) if value}
    found = []
    while occupied:
        start = occupied.pop()
        stack = [start]
        coords = []
        while stack:
            current = stack.pop()
            x, y = current % width, current // width
            coords.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    index = ny * width + nx
                    if 0 <= nx < width and 0 <= ny < height and index in occupied:
                        occupied.remove(index)
                        stack.append(index)
        xs, ys = zip(*coords)
        found.append({"area": len(coords),
                      "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1]})
    return sorted(found, key=lambda item: item["area"], reverse=True)


def similarity(a, b, roi=None):
    """RGBA visual similarity normalized by visible union area, not empty canvas."""
    if roi:
        a, b = a.crop(roi), b.crop(roi)
    black = Image.new("RGBA", a.size, (0, 0, 0, 255))
    rgb_a = Image.alpha_composite(black, a).convert("RGB")
    rgb_b = Image.alpha_composite(black, b).convert("RGB")
    differences = ImageChops.difference(rgb_a, rgb_b)
    union = ImageChops.lighter(a.getchannel("A"), b.getchannel("A"))
    bbox = union.getbbox()
    if bbox is None:
        return 1.0
    alpha_diff = ImageChops.difference(a.getchannel("A"), b.getchannel("A"))
    visible = max(1, sum(1 for value in union.tobytes() if value > 0))
    rgb_sum = sum(differences.tobytes())
    alpha_sum = sum(alpha_diff.tobytes())
    return round(max(0.0, 1.0 - (rgb_sum + alpha_sum) / (visible * 4 * 255)), 6)


def checker(size):
    result = Image.new("RGBA", size, (33, 39, 47, 255))
    draw = ImageDraw.Draw(result)
    for y in range(0, size[1], 8):
        for x in range(0, size[0], 8):
            if (x // 8 + y // 8) % 2:
                draw.rectangle((x, y, x + 7, y + 7), fill=(47, 55, 65, 255))
    return result


def displayed(im, size=None):
    # Cropping/resizing here is for a separate QA display, never the final PNG.
    if size:
        im = im.resize(size, Image.Resampling.NEAREST)
    result = checker(im.size)
    result.alpha_composite(im)
    return result


def save_contact(output, frames, arms_roi, legs_roi, frame_count):
    cell_w, cell_h = 212, 222
    contact = Image.new("RGB", (cell_w * frame_count, cell_h * 4), (20, 25, 32))
    draw = ImageDraw.Draw(contact)
    for row, direction in enumerate(DIRECTIONS):
        for index in range(frame_count):
            im = frames.get((direction, index))
            x, y = index * cell_w + 10, row * cell_h + 24
            draw.text((x, y - 18), f"{direction} {index + 1:02d}", fill="white")
            if im:
                contact.paste(displayed(im, (192, 192)).convert("RGB"), (x, y))
    contact.save(output / "contact_sheet.png")

    for direction in DIRECTIONS:
        # Columns have independent allotted widths so complete boots cannot clip.
        roi_images = [arms_roi, legs_roi]
        widths = [192] + [(roi[2] - roi[0]) * 3 for roi in roi_images]
        offsets = [10, 10 + widths[0] + 24,
                   10 + widths[0] + widths[1] + 48]
        total_width = sum(widths) + 68
        result = Image.new("RGB", (total_width, 224 * frame_count), (20, 25, 32))
        draw = ImageDraw.Draw(result)
        for index in range(frame_count):
            im = frames.get((direction, index))
            y = index * 224 + 24
            draw.text((offsets[0], y - 18), f"{direction} {index + 1:02d} full", fill="white")
            for offset, label in zip(offsets[1:], ("arms ROI", "legs ROI")):
                draw.text((offset, y - 18), label, fill="white")
            if im:
                result.paste(displayed(im, (192, 192)).convert("RGB"), (offsets[0], y))
                for offset, roi in zip(offsets[1:], roi_images):
                    crop = im.crop(roi)
                    result.paste(displayed(crop, (crop.width * 3, crop.height * 3))
                                 .convert("RGB"), (offset, y))
        result.save(output / f"{direction}_anatomy.png")


def save_preview(output, frames, args):
    # Visual copies are explicitly QA assets, separate from final input files.
    previews = output / "preview_frames"
    previews.mkdir(exist_ok=True)
    for (direction, index), im in frames.items():
        im.save(previews / f"{direction}_{index + 1:02d}.png")
    cards = "".join(
        f'<section><h2>{direction}</h2><div class="stage" data-dir="{direction}">'
        '<div class="actor"></div></div><span class="phase"></span></section>'
        for direction in DIRECTIONS)
    template = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@ACTION@ frame review</title>
<style>body{background:#131923;color:#e5eaf1;font:15px system-ui;margin:24px}
h1{font-size:22px}h2{font-size:17px}main{display:grid;grid-template-columns:repeat(2,minmax(340px,1fr));gap:16px}
section{background:#1e2734;padding:16px;border-radius:8px}.stage{height:300px;position:relative;overflow:hidden;
background:repeating-linear-gradient(0deg,#26313f 0,#26313f 1px,transparent 1px,transparent 24px)}
.actor{position:absolute;width:96px;height:96px;image-rendering:pixelated;transform-origin:48px 88px}
label{margin-right:20px}button{margin:8px}.phase{font-variant-numeric:tabular-nums}
a{color:#9bc9ff}@media(max-width:760px){main{grid-template-columns:1fr}}</style><h1>Per-frame @ACTION@ review — natural motion is not automatically approved</h1>
<p>Action: @ACTION@. Use the full loop, @COUNT@→1 seam, and complete limb images. Torso drift and similar frames are review hints.</p>
<button id="play">Pause</button><button id="step">Next frame</button><button id="reset">Reset</button>
<button id="seam">Review @COUNT@→1</button>
<label><input id="move" type="checkbox" checked>Move across the grid</label>
<label><input id="scale" type="checkbox">2× zoom</label>
<label>FPS <input id="fps" type="number" value="@FPS@" min=".1" max="16" step=".01" style="width:70px"></label>
<p>Preview speed follows @STRIDE@ px per @COUNT@-frame cycle; verify actual project movement separately.</p>
<p><a href="contact_sheet.png">Full contact sheet</a> · <a href="review.json">QA report</a></p>
<p>Limb detail: <a href="right_anatomy.png">right</a> · <a href="left_anatomy.png">left</a> ·
<a href="down_anatomy.png">down</a> · <a href="up_anatomy.png">up</a></p>
<main>@CARDS@</main><script>
const stages=[...document.querySelectorAll('.stage')];
let playing=true,phase=0,last=null;
const stride=@STRIDE@,frameCount=@COUNT@;
function render(){const frame=Math.floor(phase)%frameCount+1;const zoom=document.querySelector('#scale').checked?2:1;
for(const stage of stages){const direction=stage.dataset.dir;const actor=stage.querySelector('.actor');
const moving=document.querySelector('#move').checked;
const vertical=direction==='up'||direction==='down';
const span=Math.max(1,(vertical?stage.clientHeight:stage.clientWidth)-96*zoom-24);
const distance=moving?(phase/frameCount*stride)%span:span/2;
let x=(stage.clientWidth-96*zoom)/2,y=(stage.clientHeight-96*zoom)/2;
if(moving&&vertical)y=12+(direction==='up'?span-distance:distance);
if(moving&&!vertical)x=12+(direction==='left'?span-distance:distance);
// Integer display placement, common scale, and no added bobbing or per-frame alignment.
actor.style.left=Math.round(x)+'px';actor.style.top=Math.round(y)+'px';
actor.style.transformOrigin='0 0';actor.style.transform='scale('+zoom+')';
actor.style.backgroundImage='url(preview_frames/'+direction+'_'+String(frame).padStart(2,'0')+'.png)';
stage.parentElement.querySelector('.phase').textContent='Frame '+frame+' / '+frameCount;}}
document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play';};
document.querySelector('#step').onclick=()=>{playing=false;document.querySelector('#play').textContent='Play';phase=Math.floor(phase)+1;render();};
document.querySelector('#reset').onclick=()=>{phase=0;render();};
document.querySelector('#seam').onclick=()=>{playing=false;document.querySelector('#play').textContent='Play';phase=frameCount-1;render();};
document.querySelector('#move').onchange=render;document.querySelector('#scale').onchange=render;
function tick(now){if(last!==null&&playing)phase+=(now-last)/1000*Math.max(.1,Math.min(16,Number(document.querySelector('#fps').value||8)));
last=now;render();requestAnimationFrame(tick);}requestAnimationFrame(tick);
</script>'''
    template = template.replace("@CARDS@", cards).replace("@FPS@", str(args.fps))
    template = template.replace("@STRIDE@", str(args.stride_distance))
    template = template.replace("@COUNT@", str(args.frame_count)).replace("@ACTION@", args.action)
    (output / "preview.html").write_text(template, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, choices=(6, 8), default=8,
                        help="Final frames per direction; legacy default is 8")
    parser.add_argument("--action", choices=("walk", "run"),
                        help="Explicit action scope; no idle is required unless --require-idle is given")
    parser.add_argument("--require-idle", action="store_true",
                        help="Require four independent idle PNGs in the reviewed action scope")
    parser.add_argument("--baseline-directory", type=Path,
                        help="Optional approved original for exact per-frame change tracking")
    parser.add_argument("--alpha-threshold", type=int, default=32)
    parser.add_argument("--margins", type=rectangle, default=(4, 1, 92, 90),
                        help="Allowed opaque rectangle x0,y0,x1,y1, default 4,1,92,90")
    parser.add_argument("--anchor-roi", type=rectangle, default=(24, 18, 72, 54))
    parser.add_argument("--arms-roi", type=rectangle, default=(0, 30, 96, 72))
    parser.add_argument("--legs-roi", type=rectangle, default=(0, 56, 96, 96))
    parser.add_argument("--drift-warning", type=float, default=3.0)
    parser.add_argument("--similarity-warning", type=float, default=.985)
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--stride-distance", type=float, default=48)
    args = parser.parse_args()
    legacy_scope = args.action is None and args.frame_count == 8
    args.action = args.action or "walk"
    require_idle = args.require_idle or legacy_scope
    frame_count = args.frame_count
    atlas_size = (CANVAS[0] * frame_count, CANVAS[1] * len(DIRECTIONS))
    directory, output = args.directory.resolve(), args.output.resolve()
    if not directory.is_dir():
        parser.error("Input directory does not exist")
    if directory == output or output in directory.parents or directory in output.parents:
        parser.error("Output must be a separate QA directory, outside the asset directory and its parents")
    if not 0 <= args.alpha_threshold <= 254:
        parser.error("Alpha threshold must be 0..254")
    if not 0 <= args.similarity_warning <= 1 or args.drift_warning < 0:
        parser.error("Similarity threshold must be 0..1 and drift threshold must be nonnegative")
    if not 0 < args.fps <= 16 or args.stride_distance <= 0:
        parser.error("FPS must be 0..16 and stride distance must be positive")
    # Support existing flat assets and the accepted action/direction layout.
    action_directory = directory / args.action
    scope_directory = action_directory if action_directory.is_dir() else directory
    nested_layout = any((scope_directory / direction).is_dir() for direction in DIRECTIONS)

    def asset_path(direction, filename, root=directory):
        scope = root / args.action if (root / args.action).is_dir() else root
        grouped = any((scope / item).is_dir() for item in DIRECTIONS)
        return scope / direction / filename if grouped else scope / filename

    def relative(path):
        return os.path.relpath(path, directory)

    output.mkdir(parents=True, exist_ok=True)
    report = {"input_directory": str(directory), "read_only": True,
              "read_only_scope": "original frame, idle, atlas and manifest files; QA outputs are separate copies",
              "input_assets_read_only": True,
              "contract": {"canvas": list(CANVAS), "row_order": list(DIRECTIONS),
                           "action": args.action,
                           "layout": "action/direction" if action_directory.is_dir() else ("direction" if nested_layout else "flat"),
                           "frames_per_direction": frame_count, "atlas_size": list(atlas_size),
                           "independent_idle_required": require_idle,
                           "scope_source": "legacy walk and idle default" if legacy_scope else "explicit action; idle only if requested",
                           "allowed_alpha_rectangle": list(args.margins),
                           "margin_error_metric": "visible silhouette above alpha threshold; raw-alpha overflow is retained as a review hint",
                           "alpha_metric_threshold": args.alpha_threshold},
              "errors": [], "warnings": [], "frames": [], "directions": {},
              "manual_review": {"natural_gait": None, "identity_consistent": None,
                                "equipment_side_consistent": None,
                                "no_neighbor_image_fragments": None,
                                "in_game_verified": None,
                                "evidence": [], "per_frame": [], "pose_pairs": [],
                                "limb_identity_by_direction": {
                                    direction: {limb: blank_limb_identity() for limb in LIMBS}
                                    for direction in DIRECTIONS},
                                "limb_motion_gate": {
                                    "status": "UNREVIEWED",
                                    "each_visible_hand_and_foot_changes_position": None,
                                    "body_relative_motion_verified": None,
                                    "both_half_cycles_exchange_limbs": None,
                                    "all_frame_pairs_reviewed": None,
                                    "adjacent_and_loop_seam_reviewed": None,
                                    "per_limb": dict.fromkeys(LIMBS), "evidence": [],
                                    "unresolved_frames_or_limbs": [],
                                    "required_decision": "needs_changes or uncertain until all visible limbs have coherent, distinct positions and occlusions are explained"}}}
    frames, source_hashes = {}, {}
    for direction in DIRECTIONS:
        for index in range(frame_count):
            path = asset_path(direction, f"{direction}_{index + 1:02d}.png")
            filename = relative(path)
            if not path.exists():
                report["errors"].append(f"missing frame: {filename}")
                continue
            source_hashes[filename] = digest(path)
            im = read_png(path, report["errors"], CANVAS)
            if im is None:
                continue
            frames[(direction, index)] = im
            alpha = im.getchannel("A")
            bbox = alpha.getbbox()
            if not bbox:
                report["errors"].append(f"{filename}: fully empty")
            if alpha.getextrema()[0] != 0:
                report["errors"].append(f"{filename}: no transparent pixels")
            mask = threshold_alpha(im, args.alpha_threshold)
            solid_bbox = mask.getbbox()
            if not solid_bbox:
                report["errors"].append(f"{filename}: no visible pixels above alpha threshold {args.alpha_threshold}")
            if solid_bbox and (solid_bbox[0] < args.margins[0] or solid_bbox[1] < args.margins[1]
                               or solid_bbox[2] > args.margins[2] or solid_bbox[3] > args.margins[3]):
                report["errors"].append(f"{filename}: visible silhouette crosses margin; alpha>{args.alpha_threshold}, bbox={solid_bbox}")
            outside_mask = Image.new("L", CANVAS, 255)
            ImageDraw.Draw(outside_mask).rectangle((args.margins[0], args.margins[1],
                                                    args.margins[2] - 1, args.margins[3] - 1), fill=0)
            outside_alpha = ImageChops.multiply(alpha, outside_mask)
            outside_pixels = sum(outside_alpha.histogram()[1:])
            outside_max_alpha = outside_alpha.getextrema()[1]
            if outside_pixels and outside_max_alpha <= args.alpha_threshold:
                report["warnings"].append(
                    f"{filename}: {outside_pixels} low-alpha pixel(s) outside margin, max alpha={outside_max_alpha}; "
                    "inspect neutral-background composite, do not rewrite approved PNGs just to clear this hint")
            parts = components(mask)
            tiny = [part for part in parts[1:] if part["area"] <= 8]
            if tiny:
                report["warnings"].append(f"{filename}: {len(tiny)} detached tiny component(s); inspect for neighbor fragments")
            anchor = centroid(mask.crop(args.anchor_roi))
            if anchor:
                anchor = [round(anchor[0] + args.anchor_roi[0], 3),
                          round(anchor[1] + args.anchor_roi[1], 3)]
            record = {"file": filename, "sha256": source_hashes[filename],
                      "alpha_bbox": bbox, "solid_bbox": solid_bbox,
                      "outside_margin_alpha": {"pixel_count": outside_pixels, "max_alpha": outside_max_alpha},
                      "visible_height_including_equipment": solid_bbox[3] - solid_bbox[1] if solid_bbox else 0,
                      "foot_or_lowest_equipment_y": solid_bbox[3] - 1 if solid_bbox else None,
                      "anchor_roi_centroid": anchor, "components": parts}
            if args.baseline_directory:
                old = asset_path(direction, path.name, args.baseline_directory.resolve())
                if old.exists():
                    record["unchanged_from_baseline"] = source_hashes[filename] == digest(old)
                else:
                    record["unchanged_from_baseline"] = None
            report["frames"].append(record)
            report["manual_review"]["per_frame"].append({
                "file": filename, "frame": index + 1, "action": args.action,
                "direction": direction, "source_sha256": source_hashes[filename],
                "limb_identity_map": f"manual_review.limb_identity_by_direction.{direction}",
                "torso_origin_native_xy": None, "torso_anchor_visual_identifier": "",
                "limbs": {limb: blank_limb_observation(limb) for limb in LIMBS},
                "previous_frame": (index - 1) % frame_count + 1,
                "transition_from_previous": "",
                "half_cycle_frame": (index + frame_count // 2) % frame_count + 1,
                "half_cycle_identity_exchange_observation": "",
                "arm_observation": "", "leg_A_observation": "",
                "leg_B_observation": "", "support_leg": "", "swing_leg": "",
                "phase": "", "next_frame": (index + 1) % frame_count + 1,
                "transition_to_next_including_loop_seam": "",
                **({"transition_to_next_including_08_to_01": ""} if frame_count == 8 else {}),
                "equipment_observation": "", "keep_or_repair": "", "accepted": None})

        idle = asset_path(direction, f"{direction}_idle.png")
        if require_idle and not idle.exists():
            report["errors"].append(f"missing independent idle: {relative(idle)}")
        elif require_idle:
            source_hashes[relative(idle)] = digest(idle)
            im = read_png(idle, report["errors"], CANVAS)
            if im is not None:
                idle_alpha = im.getchannel("A")
                idle_bbox = threshold_alpha(im, args.alpha_threshold).getbbox()
                if not idle_bbox:
                    report["errors"].append(f"{idle.name}: independent idle has no pixels above alpha threshold")
                elif (idle_bbox[0] < args.margins[0] or idle_bbox[1] < args.margins[1]
                      or idle_bbox[2] > args.margins[2] or idle_bbox[3] > args.margins[3]):
                    report["errors"].append(f"{idle.name}: visible silhouette crosses margin")
                raw_idle_bbox = idle_alpha.getbbox()
                if raw_idle_bbox and (raw_idle_bbox[0] < args.margins[0] or raw_idle_bbox[1] < args.margins[1]
                                      or raw_idle_bbox[2] > args.margins[2] or raw_idle_bbox[3] > args.margins[3]):
                    report["warnings"].append(f"{idle.name}: raw alpha crosses margin; inspect composite alongside threshold silhouette")
        records = [record for record in report["frames"] if Path(record["file"]).name.startswith(direction + "_")]
        heights = [record["visible_height_including_equipment"] for record in records]
        anchors = [record["anchor_roi_centroid"] for record in records if record["anchor_roi_centroid"]]
        stats = {"duplicate_pairs": [], "near_pairs": [], "all_pair_similarity": [],
                 "adjacent_similarity": [], "expected_all_pair_count": frame_count * (frame_count - 1) // 2,
                 "similarity_scope": "raw fixed image ROIs; not torso-registered joints or detected limb motion",
                 "height_range_including_equipment": [min(heights), max(heights)] if heights else None}
        if heights and max(heights) - min(heights) > args.drift_warning:
            report["warnings"].append(f"{direction}: visible height range {max(heights)-min(heights)}px; distinguish equipment/swing from body stretch")
        if anchors:
            reference = [median(p[0] for p in anchors), median(p[1] for p in anchors)]
            drift = max(math.dist(point, reference) for point in anchors)
            stats["anchor_roi_max_deviation_from_median"] = round(drift, 3)
            if drift > args.drift_warning:
                report["warnings"].append(f"{direction}: anchor ROI centroid deviation {drift:.2f}px; inspect true body drift")
        for a in range(frame_count):
            if (direction, a) not in frames:
                continue
            for b in range(a + 1, frame_count):
                if (direction, b) not in frames:
                    continue
                first, second = frames[(direction, a)], frames[(direction, b)]
                pair = [a + 1, b + 1]
                if first.tobytes() == second.tobytes():
                    stats["duplicate_pairs"].append(pair)
                    report["errors"].append(f"{direction}: exact duplicate frames {pair}")
                scores = {"frames": pair, "whole": similarity(first, second),
                          "arms": similarity(first, second, args.arms_roi),
                          "legs": similarity(first, second, args.legs_roi)}
                stats["all_pair_similarity"].append(scores)
                source_files = [{"file": relative(asset_path(direction, f"{direction}_{i:02d}.png")),
                                 "sha256": source_hashes[relative(asset_path(direction, f"{direction}_{i:02d}.png"))]}
                                for i in pair]
                report["manual_review"]["pose_pairs"].append(
                    blank_pair_review(direction, pair, frame_count, source_files))
                if any(scores[part] >= args.similarity_warning for part in ("whole", "arms", "legs")):
                    stats["near_pairs"].append(scores)
            following = (a + 1) % frame_count
            if (direction, following) in frames:
                stats["adjacent_similarity"].append({
                    "frames": [a + 1, following + 1],
                    "whole": similarity(frames[(direction, a)], frames[(direction, following)]),
                    "arms": similarity(frames[(direction, a)], frames[(direction, following)], args.arms_roi),
                    "legs": similarity(frames[(direction, a)], frames[(direction, following)], args.legs_roi)})
        if stats["near_pairs"]:
            report["warnings"].append(f"{direction}: {len(stats['near_pairs'])} similar whole/arm/leg pair(s); inspect whether pose progression is readable")
        report["directions"][direction] = stats

    atlas_path = directory / f"{args.action}_atlas.png"
    if not atlas_path.exists() and scope_directory != directory:
        atlas_path = scope_directory / f"{args.action}_atlas.png"
    if not atlas_path.exists() and nested_layout and directory.name == args.action:
        atlas_path = directory.parent / f"{args.action}_atlas.png"
    if not atlas_path.exists():
        report["errors"].append(f"missing {args.action}_atlas.png")
    else:
        source_hashes[relative(atlas_path)] = digest(atlas_path)
        atlas = read_png(atlas_path, report["errors"], atlas_size)
        if atlas is not None:
            for row, direction in enumerate(DIRECTIONS):
                for index in range(frame_count):
                    im = frames.get((direction, index))
                    if im and atlas.crop((index * 96, row * 96, (index + 1) * 96, (row + 1) * 96)).tobytes() != im.tobytes():
                        report["errors"].append(f"atlas mismatch: {direction}_{index + 1:02d}.png")
    manifest_path = scope_directory / "manifest.json"
    if manifest_path.exists():
        source_hashes[relative(manifest_path)] = digest(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest root must be an object")
            report["manifest"] = manifest
            if manifest.get("row_order") and manifest["row_order"] != list(DIRECTIONS):
                report["errors"].append("manifest row_order differs from atlas contract")
            if manifest.get("canvas") and manifest["canvas"] != list(CANVAS):
                report["errors"].append("manifest canvas differs from frame contract")
            if manifest.get("sheet_size") and manifest["sheet_size"] != list(atlas_size):
                report["errors"].append("manifest sheet_size differs from atlas contract")
            if manifest.get("directions") and manifest["directions"] != dict.fromkeys(DIRECTIONS, frame_count):
                report["errors"].append(f"manifest directions differ from four-direction {frame_count}-frame contract")
            if require_idle and manifest.get("independent_idle_included") is False:
                report["warnings"].append("manifest claims no independent idle; inspect actual files and engine load")
        except (ValueError, TypeError) as error:
            report["errors"].append(f"manifest unreadable: {error}")
    report["mechanical_pass"] = not report["errors"]
    report["natural_gait_status"] = "UNVERIFIED_REQUIRES_MANUAL_PHASE_AND_MOVEMENT_REVIEW"
    report["limb_motion_status"] = "UNVERIFIED_REQUIRES_FOUR_LIMB_AND_ALL_PAIR_REVIEW"
    save_contact(output, frames, args.arms_roi, args.legs_roi, frame_count)
    save_preview(output, frames, args)
    changed = [name for name, old_hash in source_hashes.items()
               if digest(directory / name) != old_hash]
    if changed:
        raise RuntimeError(f"Input changed during review (possibly another process): {changed}")
    report["input_hashes_unchanged_during_run"] = True
    (output / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Read-only {args.action} review", "", f"Input: `{directory}`", "",
             f"Scope: {frame_count} frames per direction; independent idle required: {require_idle}",
             f"Mechanical pass: {report['mechanical_pass']}",
             "Natural gait: UNVERIFIED — fill per-frame observations and verify actual movement.", "",
             "Four-limb motion: UNVERIFIED — the manual limb_motion_gate is UNREVIEWED.",
             f"Review every arm and leg separately in all {frame_count * (frame_count - 1) // 2} pairs per direction, plus ordered adjacent transitions and the loop seam.",
             "Torso bob, texture changes and arbitrary pixel jitter do not prove hand or foot movement.", "",
             "## Structural errors", ""]
    lines += [f"- {error}" for error in report["errors"]] or ["None."]
    lines += ["", "## Review hints", ""]
    lines += [f"- {warning}" for warning in report["warnings"]] or ["None. This does not certify natural motion."]
    lines += ["", "All-pair ROI scores are raw image similarity hints, not detected anatomical phases or torso-relative motion.",
              "All display images preserve the full source canvas; ROI crops are labelled and only used for review.",
              "Input PNGs are never rewritten. Joint positions, limb identities, pair decisions and gate fields remain unreviewed for visual inspection.", ""]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"mechanical_pass": report["mechanical_pass"],
                      "errors": len(report["errors"]), "warnings": len(report["warnings"]),
                      "natural_gait": "UNVERIFIED", "limb_motion": "UNVERIFIED",
                      "report": str(output / "review.json")}, ensure_ascii=False))
    return 0 if report["mechanical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
