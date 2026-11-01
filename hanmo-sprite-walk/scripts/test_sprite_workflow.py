#!/usr/bin/env python3
"""Synthetic isolated workflow checks, not artistic or gameplay acceptance."""
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw, PngImagePlugin
import sprite_workflow as sw


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sprite-workflow-fixture-")
        self.base = Path(self.temp.name)
        self.master = self.base / "natural.png"
        im = Image.new("RGBA", (96, 96))
        ImageDraw.Draw(im).rectangle((40, 14, 53, 85), fill=(153, 64, 72, 255))
        im.save(self.master)
        self.source = self.base / "sequence.png"
        self.sheet()
        self.mapping = self.base / "mapping.json"
        self.spec = {"cell_size": [96, 96], "origin": [0, 0],
                     "column_pitch": 96, "row_pitch": 96, "cols": 3,
                     "evidence": "Synthetic fixture measured six full transparent cells; no visual claim"}
        self.mapping.write_text(json.dumps(self.spec))
        self.job = self.init(["right", "left"], ["walk", "run"])

    def tearDown(self):
        self.temp.cleanup()

    def call(self, command):
        args = sw.parser().parse_args(command)
        return args.func(args)

    def init(self, directions, actions):
        return Path(self.call(["init", "--reference", str(self.master), "--output", str(self.base / "jobs"),
                               "--name", "测试角色", "--sequence-mode", "legacy_embedded_stand",
                               "--directions", *directions, "--actions", *actions])["job"])

    def set_master(self, direction="right"):
        return self.call(["set-master", "--job", str(self.job), "--direction", direction,
                          "--source", str(self.master), "--evidence", "Observed fixture stand, relaxed arms, naturally closed feet",
                          "--closed-feet-verified", "--reference-direction"])

    def ingest(self, task="right-walk"):
        return self.call(["ingest", "--job", str(self.job), "--task", task,
                          "--source", str(self.source), "--mapping", str(self.mapping)])

    def sheet(self, duplicate=False, first_matches_master=False):
        sheet = Image.new("RGBA", (288, 192))
        for i in range(6):
            tile = Image.new("RGBA", (96, 96))
            n = 1 if duplicate and i == 2 else i
            ImageDraw.Draw(tile).rectangle((22+n*3, 15+n, 47+n*2, 83-n), fill=(30+n*18, 83, 137, 255))
            if first_matches_master and i == 1:
                tile = Image.open(self.master).convert("RGBA")
            sheet.paste(tile, ((i%3)*96, (i//3)*96))
        sheet.save(self.source)

    def test_default_plan_is_pending_and_versioned(self):
        job2 = self.init(["right"], ["walk"])
        self.assertNotEqual(self.job, job2)
        manifest = sw.read_json(job2 / "manifest.json")
        self.assertEqual(manifest["actions"], ["walk"])
        self.assertEqual(manifest["delivery_status"], "partial")
        self.assertEqual(manifest["tasks"][0]["provider_status"], "provider_pending")
        self.assertEqual(manifest["tasks"][0]["status"], "source_required")
        self.assertEqual(manifest["masters"]["right"]["status"], "source_required")
        prompt = (job2 / manifest["tasks"][0]["prompt"]).read_text()
        self.assertIn("ORIGINAL reference", prompt)
        self.assertIn("Never mirror", prompt)

    def test_action_specific_prompt_avoids_mixed_phase_rules(self):
        manifest = sw.read_json(self.job / "manifest.json")
        prompts = {t["action"]: (self.job/t["prompt"]).read_text() for t in manifest["tasks"]}
        self.assertIn("WALK only:", prompts["walk"])
        self.assertNotIn("RUN only:", prompts["walk"])
        self.assertIn("RUN only:", prompts["run"])
        self.assertNotIn("WALK only:", prompts["run"])

    def test_requires_observed_master_and_transparency(self):
        with self.assertRaisesRegex(sw.WorkflowError, "set-master"):
            self.ingest()
        with self.assertRaisesRegex(sw.WorkflowError, "closed-feet"):
            self.call(["set-master", "--job", str(self.job), "--direction", "right", "--source", str(self.master), "--evidence", "not observed"])
        self.set_master()
        Image.new("RGBA", (288,192), (255,255,255,255)).save(self.source)
        with self.assertRaisesRegex(sw.WorkflowError, "Opaque backgrounds"):
            self.ingest()

    def test_out_of_bounds_and_non_common_mapping_rejected(self):
        self.set_master()
        self.spec["origin"] = [2, 0]
        self.mapping.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(sw.WorkflowError, "out of bounds"):
            self.ingest()
        self.spec["origin"] = [0,0]
        self.spec["per_frame_boxes"] = []
        self.mapping.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(sw.WorkflowError, "Unsupported mapping"):
            self.ingest()
        del self.spec["per_frame_boxes"]
        self.spec["evidence"] = " "
        self.mapping.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(sw.WorkflowError, "cannot be empty"):
            self.ingest()

    def test_activity_duplicates_archived_rejected_retry_is_new(self):
        self.set_master()
        self.sheet(duplicate=True)
        result = self.ingest()
        self.assertFalse(result["accepted"])
        manifest = sw.read_json(self.job / "manifest.json")
        task = manifest["tasks"][0]
        self.assertIsNone(task["active_attempt"])
        archived = self.job / task["attempts"][0]["source"]
        self.assertEqual(archived.read_bytes(), self.source.read_bytes())
        previous = sw.sha(archived)
        self.sheet()
        result2 = self.ingest()
        self.assertEqual(result2["attempt"], 2)
        self.assertTrue(result2["accepted"])
        self.assertEqual(sw.sha(archived), previous)

    def test_master_repeat_warns_but_keeps_evidence(self):
        self.set_master()
        self.sheet(first_matches_master=True)
        result = self.ingest()
        self.assertTrue(result["accepted"])
        self.assertIn("master_repeated_in_activity", [w["code"] for w in result["warnings"]])
        self.assertEqual(result["structure"], "needs_review")

    def test_cross_cell_and_crop_edge_warn_without_cleanup(self):
        self.set_master()
        self.spec["column_pitch"] = 88
        self.mapping.write_text(json.dumps(self.spec))
        with Image.open(self.source) as sheet:
            im = sheet.convert("RGBA")
        ImageDraw.Draw(im).rectangle((88, 12, 95, 16), fill=(220,20,30,255))
        im.save(self.source)
        result = self.ingest()
        codes = [w["code"] for w in result["warnings"]]
        self.assertIn("crop_edge_alpha", codes)
        self.assertIn("shared_crop_alpha", codes)
        self.assertEqual(result["structure"], "needs_review")
        self.assertEqual(sw.sha(self.source), sw.read_json(self.job / "manifest.json")["tasks"][0]["attempts"][0]["source_sha256"])

    def test_master_exact_bytes_and_versioned_retry(self):
        self.set_master()
        self.ingest()
        old = sw.read_json(self.job / "manifest.json")["tasks"][0]["attempts"][0]
        self.assertEqual((self.job / old["frames"][0]["path"]).read_bytes(), self.master.read_bytes())
        self.assertEqual((self.job / old["path"] / "stand.png").read_bytes(), self.master.read_bytes())
        self.ingest("right-run")
        data = sw.read_json(self.job / "manifest.json")
        run = next(t for t in data["tasks"] if t["id"] == "right-run")["attempts"][0]
        self.assertEqual((self.job / run["frames"][0]["path"]).read_bytes(), self.master.read_bytes())
        retry = self.ingest()
        self.assertEqual(retry["attempt"], 2)
        self.assertEqual(sw.sha(self.job / old["frames"][0]["path"]), old["frames"][0]["sha256"])

    def test_hash_protects_reference_master_and_received_frames(self):
        self.set_master()
        self.ingest()
        _, data = sw.load_job(self.job)
        protected = [data["reference"]["path"], data["masters"]["right"]["path"], data["tasks"][0]["attempts"][0]["frames"][1]["path"]]
        for relative in protected:
            path = self.job / relative
            original = path.read_bytes()
            path.write_bytes(original + b"corrupt")
            with self.assertRaisesRegex(sw.WorkflowError, "missing or changed"):
                self.call(["finalize", "--job", str(self.job)])
            path.write_bytes(original)
        sw.load_job(self.job)

    def test_partial_delivery_self_contained_and_row_order(self):
        self.set_master("left")
        self.ingest("left-walk")
        delivery = self.call(["finalize", "--job", str(self.job)])
        self.assertEqual(delivery["delivery_status"], "partial")
        snapshot = sw.read_json(delivery["manifest"])
        atlas = snapshot["snapshot_atlases"][0]
        self.assertEqual([r["direction"] for r in atlas["rows"]], ["right", "left"])
        self.assertTrue(atlas["missing_rows_are_transparent"])
        with Image.open(Path(delivery["manifest"]).parent / atlas["path"]) as im:
            self.assertEqual(im.size, (576,192))
            self.assertIsNone(im.crop((0,0,576,96)).getchannel("A").getbbox())
            self.assertIsNotNone(im.crop((0,96,576,192)).getchannel("A").getbbox())
        preview = Path(delivery["preview"]).read_text()
        self.assertIn("data:image/png;base64,", preview)
        self.assertIn("含起步全帧", preview)
        self.assertIn("循环范围未核验", preview)
        self.assertNotIn("__DATA__", preview)
        self.assertEqual(snapshot["review"]["visual"], "needs_review")
        again = self.call(["finalize", "--job", str(self.job)])
        self.assertNotEqual(again["preview"], delivery["preview"])
        self.assertTrue(Path(delivery["preview"]).is_file())

    def test_all_ingested_is_candidate_not_visual_acceptance(self):
        self.job = self.init(["up"], ["walk"])
        self.set_master("up")
        self.spec["loop_indices"] = [2,3,4,5,6]
        self.mapping.write_text(json.dumps(self.spec))
        self.ingest("up-walk")
        result = self.call(["finalize", "--job", str(self.job)])
        self.assertEqual(result["delivery_status"], "all_tasks_ingested_candidates")
        manifest = sw.read_json(result["manifest"])
        self.assertEqual(manifest["review"]["visual"], "needs_review")
        self.assertEqual(manifest["review"]["user"], "not_reviewed")
        attempt = manifest["tasks"][0]["attempts"][0]
        observations = self.job / attempt["observations"]
        evidence = sw.read_json(observations)
        self.assertTrue(all(v is None for v in evidence["limb_review"].values()))
        self.assertEqual(len(evidence["pairwise_observations"]),15)
        evidence["frame_observations"][0]["identity"] = "independently observed fixture only"
        observations.write_text(json.dumps(evidence))
        sw.load_job(self.job)  # observer template is editable, artwork is immutable

    def test_generation_record_pending_or_explicit_fixture(self):
        self.set_master()
        first = self.ingest()
        self.assertEqual(first["generation_provenance"], "pending")
        record = self.base / "generation.json"
        record.write_text(json.dumps({"kind": "fixture_replay", "tool": "offline_fixture",
            "model": "none_fixture", "prompt": "Synthetic rectangles, not generated artwork",
            "attachments": [{"path": str(self.master), "role": "direction_master_fixture"}],
            "executed_at": "2026-10-20T12:00:00+08:00"}))
        result = self.call(["ingest", "--job", str(self.job), "--task", "right-walk", "--source", str(self.source),
                           "--mapping", str(self.mapping), "--generation-record", str(record)])
        self.assertEqual(result["generation_provenance"], "fixture_replay")
        data = sw.read_json(self.job/"manifest.json")
        attempt = data["tasks"][0]["attempts"][1]
        archived = self.job/attempt["generation_record"]
        self.assertEqual(archived.read_bytes(), record.read_bytes())
        self.assertIn(str(archived.relative_to(self.job)), data["protected_files"])
        self.assertFalse(sw.read_json(self.job/attempt["path"]/"evidence.json")["generation_claims_verified_by_script"])
        payload = sw.read_json(record)
        payload.update(kind="tool_generation",provider_kind="external")
        record.write_text(json.dumps(payload))
        with self.assertRaisesRegex(sw.WorkflowError, "external_task_id"):
            self.call(["ingest", "--job", str(self.job), "--task", "right-walk", "--source", str(self.source),
                       "--mapping", str(self.mapping), "--generation-record", str(record)])

    def test_preview_buttons_and_first_play_advance_from_visible_frame(self):
        # Execute the script from an actual new delivery, rather than matching
        # implementation strings or reimplementing advance() in Python.
        node = shutil.which("node") or ""
        if not Path(node).is_file():
            self.skipTest("Node is required for the executable preview UI regression")
        self.set_master("right")
        self.ingest("right-walk")
        self.set_master("left")
        self.spec["loop_indices"] = [2, 3, 4, 5, 6]
        self.mapping.write_text(json.dumps(self.spec))
        self.ingest("left-walk")
        manifest = sw.read_json(self.job / "manifest.json")
        right = next(t for t in manifest["tasks"] if t["id"] == "right-walk")
        observation_path = self.job / right["attempts"][0]["observations"]
        observation = sw.read_json(observation_path)
        observation["status"] = "needs_changes"
        observation_path.write_text(json.dumps(observation))
        result = self.call(["finalize", "--job", str(self.job)])
        harness = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const page = fs.readFileSync(process.argv[1], 'utf8');
const script = page.split('<script>')[1].split('</script>')[0];
const expectFailure = process.argv[2] === 'needs_changes';
const elements = new Map(), raf = [], draws = [], resizeCallbacks=[], windowEvents={};
const defaults = {fps:'8', scale:'3', stride:'48', mode:'inplace', loop:'1,2,3,4,5,6', loopMode:'full'};
class Element {
  constructor(id='') { this.id=id; this.value=defaults[id]||''; this.textContent=''; this.children=[];this.style={};this.clientWidth=1000;this.clientHeight=330; }
  replaceChildren(...children) {
    this.children=children;
    if (children[0] && !children.some(c=>c.value===this.value)) this.value=children[0].value;
  }
  append(...children) { this.children.push(...children); }
  getContext() { return {clearRect(){},drawImage(image,...args){draws.push({source:image.src,args});}}; }
}
const document = {
  getElementById(id) { if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id); },
  createElement() { return new Element(); }
};
class MockImage {
  set src(value) { this._src=value; queueMicrotask(()=>this.onload()); }
  get src() { return this._src; }
  cloneNode() { const image=new MockImage();image._src=this.src;return image; }
}
class MockResizeObserver {constructor(callback){resizeCallbacks.push(callback)}observe(target){this.target=target}}
const context=vm.createContext({document,Image:MockImage,ResizeObserver:MockResizeObserver,window:{addEventListener(name,callback){windowEvents[name]=callback}},requestAnimationFrame(fn){raf.push(fn);}});
vm.runInContext(script,context);
const get=id=>document.getElementById(id);
const click=id=>get(id).onclick();
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const tick=time=>{assert(raf.length);raf.shift()(time);};
function shown(expected,label) {
  const match=/ · F(\d+) \/ 6/.exec(get('info').textContent);
  assert(match, 'Visible frame info is missing');
  assert.equal(Number(match[1]),expected,label);
  const images=vm.runInContext('current.images',context);
  assert.equal(draws.at(-1).source,images[expected-1],label+' draws actual corresponding PNG');
}
(async()=>{
  await flush();
  shown(1,'initial F1');
  const expectedVisual=expectFailure?'needs_changes':'needs_review';
  assert(get('status').textContent.includes('视觉 '+expectedVisual),'global status shows actual conservative visual result');
  assert.equal(get('status').className,expectFailure?'failed':'warn','failure has a prominent global style');
  assert(get('status').textContent.includes('用户未认可'),'negative/positive observations never approve user status');
  assert(get('taskStatus').textContent.includes('视觉 '+expectedVisual),'selected task shows own observation status');
  assert.equal(get('taskStatus').className,expectFailure?'failed':'warn','selected rejected task has prominent failure style');
  assert.equal(resizeCallbacks.length,1,'viewport resize observer installed');
  get('scale').value='1';get('scale').onchange();
  function sized(width,height,callback){
    get('stageViewport').clientWidth=width;get('stageViewport').clientHeight=height;callback();
    assert.equal(get('stage').width,width,'backing matches content CSS width');
    assert.equal(get('stage').height,height,'backing matches content CSS height');
    assert.equal(get('stage').style.width,width+'px','CSS width explicitly matches backing');
    assert.equal(get('stage').style.height,height+'px','CSS height explicitly matches backing');
    assert.equal(draws.at(-1).args[2],96,'scale 1 draws exactly 96 CSS pixels wide');
    assert.equal(draws.at(-1).args[3],96,'scale 1 draws exactly 96 CSS pixels high');
  }
  sized(1076,330,resizeCallbacks[0]);
  sized(320,280,resizeCallbacks[0]);
  sized(768,328,windowEvents.resize);
  get('scale').value='3';get('scale').onchange();
  assert.equal(draws.at(-1).args[2],288,'scale 3 uses integer 288 CSS pixels');
  assert.equal(draws.at(-1).args[3],288,'scale 3 preserves aspect ratio');
  assert.equal(get('frames').children.length,6,'all six review thumbnails');
  click('step');shown(2,'first step moves F1 to F2');
  click('step');shown(3,'next step moves F2 to F3');
  click('stand');shown(1,'stand reset restores F1');
  click('step');shown(2,'first step after reset moves to F2');
  click('stand');click('play');
  tick(1000);tick(1062.5);shown(1,'F1 is held only until first frame interval');
  tick(1125);shown(2,'first playback interval advances to F2, not repeated F1');
  tick(1187.5);tick(1250);shown(3,'second playback interval advances to F3');
  click('play');click('stand');
  get('loop').value='2,4,6';click('apply');shown(1,'applying custom loop retains stand');
  click('step');shown(2,'current stand outside custom loop selects first member');
  click('step');shown(4,'custom loop moves to next actual member');
  click('step');shown(6,'custom loop skips no declared member');
  click('step');shown(2,'custom loop wraps to first member');
  get('loop').value='3,1,5';click('apply');shown(1,'custom loop reset to stand');
  click('step');shown(5,'current frame inside custom loop advances relative to current position');
  click('step');shown(3,'out-of-order custom loop wraps correctly');
  get('loopMode').value='full';get('loopMode').onchange();shown(1,'full loop mode resets to stand');
  click('step');shown(2,'first full-loop step after mode change is F2');
  get('direction').value='left';await get('direction').onchange();shown(1,'direction selection resets to F1');
  assert(get('taskStatus').textContent.includes('视觉 needs_review'),'clean selected group is not tainted by other task');
  assert.equal(get('taskStatus').className,'warn','clean selected group is still only a candidate');
  assert(get('status').textContent.includes('视觉 '+expectedVisual),'global negative state persists when selecting another group');
  click('step');shown(2,'declared loop excludes F1 and begins with F2');
  click('stand');click('play');tick(2000);tick(2062.5);tick(2125);
  shown(2,'first play after direction/reset obeys custom loop');
  console.log('PREVIEW_EXECUTION_PASS: initial/reset/select/apply/full/custom/first-play/resize/css-pixels');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        for observation_status, visual in (("needs_changes", "needs_changes"), ("approved", "needs_review"), ("passed", "needs_review")):
            observation["status"] = observation_status
            observation_path.write_text(json.dumps(observation))
            result = self.call(["finalize", "--job", str(self.job)])
            self.assertEqual(result["visual"], visual)
            self.assertEqual(result["user"], "not_reviewed")
            snapshot = sw.read_json(result["manifest"])
            self.assertEqual(snapshot["review"]["visual"], visual)
            self.assertEqual(snapshot["review"]["user"], "not_reviewed")
            executed = subprocess.run([node, "-e", harness, result["preview"], visual], capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertIn("PREVIEW_EXECUTION_PASS", executed.stdout)

    def test_only_active_negative_observation_lowers_visual_status(self):
        self.set_master()
        self.ingest()
        data = sw.read_json(self.job/"manifest.json")
        old = self.job/data["tasks"][0]["attempts"][0]["observations"]
        observation = sw.read_json(old)
        observation["status"] = "needs_changes"
        old.write_text(json.dumps(observation))
        failure = self.call(["finalize", "--job", str(self.job)])
        self.assertEqual(failure["visual"], "needs_changes")
        self.sheet(duplicate=True)
        self.assertFalse(self.ingest()["accepted"])
        data = sw.read_json(self.job/"manifest.json")
        rejected = self.job/data["tasks"][0]["attempts"][1]["observations"]
        observation = sw.read_json(rejected)
        observation["status"] = "needs_changes"
        rejected.write_text(json.dumps(observation))
        self.sheet()
        self.assertTrue(self.ingest()["accepted"])
        latest = self.call(["finalize", "--job", str(self.job)])
        self.assertEqual(latest["visual"], "needs_review")
        self.assertEqual(latest["user"], "not_reviewed")
        snapshot = sw.read_json(latest["manifest"])
        self.assertEqual(snapshot["review"]["visual"], "needs_review")
        task = snapshot["tasks"][0]
        self.assertEqual(task["active_attempt"], 3)
        self.assertEqual(task["review"]["visual"], "needs_review")
        self.assertEqual(task["attempts"][0]["independent_observations"]["status"], "needs_changes")
        self.assertEqual(task["attempts"][1]["independent_observations"]["status"], "needs_changes")

    def test_motion_reference_archived_with_identity_separation(self):
        result = self.call(["init", "--reference", str(self.master), "--motion-reference", str(self.source),
                            "--output", str(self.base / "motion-jobs"), "--name", "动作参考"])
        root, manifest = sw.load_job(result["job"])
        motion = manifest["motion_reference"]
        self.assertEqual((root / motion["path"]).read_bytes(), self.source.read_bytes())
        self.assertEqual(motion["role"], "motion_only_not_character_identity")
        self.assertIn("never character identity", (root / manifest["tasks"][0]["prompt"]).read_text())

    def new_separate_job(self, actions=("walk",)):
        self.job = Path(self.call(["init", "--reference", str(self.master), "--output", str(self.base / "new-jobs"),
                                  "--name", "完整活动循环", "--directions", "right", "--actions", *actions])["job"])
        return self.job

    def test_new_jobs_default_to_six_active_and_separate_stand(self):
        self.new_separate_job()
        manifest = sw.read_json(self.job / "manifest.json")
        self.assertEqual(manifest["sequence_mode"], "separate_stand")
        self.assertEqual(manifest["presentation_layout"], "stand_first7")
        prompt = (self.job / manifest["tasks"][0]["prompt"]).read_text()
        self.assertIn("All six cells are ACTIVE", prompt)
        self.assertNotIn("only cells 2-6 provide movement", prompt)
        self.set_master()
        self.ingest()
        manifest = sw.read_json(self.job / "manifest.json")
        attempt = manifest["tasks"][0]["attempts"][0]
        self.assertEqual(attempt["mapping"]["loop_indices"], [1, 2, 3, 4, 5, 6])
        with Image.open(self.source) as source:
            for n, rec in enumerate(attempt["frames"]):
                with Image.open(self.job / rec["path"]) as output:
                    expected = source.crop(((n % 3)*96, (n // 3)*96, (n % 3+1)*96, (n // 3+1)*96))
                    self.assertEqual(output.tobytes(), expected.tobytes(), f"Active {n+1} must survive ingest")
                self.assertEqual(rec["provenance"], "shared_canvas_nearest_mapping")
                self.assertIsNotNone(rec["source_crop"])
        attempt_dir = self.job / attempt["path"]
        self.assertFalse((attempt_dir / "generated_slot1_unused.png").exists())
        for name in ("stand.png", "start.png"):
            self.assertEqual((attempt_dir / name).read_bytes(), self.master.read_bytes())
        with Image.open(attempt_dir / "legacy7_atlas.png") as wrapper:
            self.assertEqual(wrapper.size, (672, 96))
            with Image.open(self.master) as master:
                self.assertEqual(wrapper.crop((0,0,96,96)).tobytes(), master.tobytes())
            for n, rec in enumerate(attempt["frames"]):
                with Image.open(self.job / rec["path"]) as active:
                    self.assertEqual(wrapper.crop(((n+1)*96,0,(n+2)*96,96)).tobytes(), active.tobytes())

    def test_first_active_duplicate_is_detected_in_new_mode(self):
        self.new_separate_job()
        self.set_master()
        with Image.open(self.source) as image:
            source = image.copy()
        source.paste(source.crop((0,0,96,96)), (192,96))
        source.save(self.source)
        result = self.ingest()
        self.assertFalse(result["accepted"])
        duplicate = next(w for w in result["warnings"] if w["code"] == "duplicate_active_frames")
        self.assertEqual(duplicate["pairs"], [[1, 6]])
        task = sw.read_json(self.job / "manifest.json")["tasks"][0]
        self.assertIsNone(task["active_attempt"])
        self.assertEqual(len(task["attempts"][0]["frames"]), 6)

    def test_separate_ingest_rejects_missing_or_reordered_active_cycle(self):
        self.new_separate_job()
        self.set_master()
        for loops in ([2,3,4,5,6], [2,1,3,4,5,6], [6,5,4,3,2,1]):
            with self.subTest(loop_indices=loops):
                self.spec["loop_indices"] = loops
                self.mapping.write_text(json.dumps(self.spec))
                with self.assertRaisesRegex(sw.WorkflowError, "requires complete loop_indices"):
                    self.ingest()
                task = sw.read_json(self.job / "manifest.json")["tasks"][0]
                self.assertEqual(task["attempts"], [])
                self.assertIsNone(task["active_attempt"])
        self.spec["loop_indices"] = [1,2,3,4,5,6]
        self.mapping.write_text(json.dumps(self.spec))
        self.assertTrue(self.ingest()["accepted"])

    def test_legacy_ingest_keeps_explicit_partial_or_reordered_loops(self):
        self.set_master()
        for loops in ([2,3,4,5,6], [6,4,2]):
            self.spec["loop_indices"] = loops
            self.mapping.write_text(json.dumps(self.spec))
            self.assertTrue(self.ingest()["accepted"])
            attempt = sw.read_json(self.job / "manifest.json")["tasks"][0]["attempts"][-1]
            self.assertEqual(attempt["mapping"]["loop_indices"], loops)

    def test_master_repeat_in_first_active_is_observed_not_discarded(self):
        self.new_separate_job()
        self.set_master()
        with Image.open(self.source) as image:
            source = image.copy()
        with Image.open(self.master) as master:
            source.paste(master, (0,0))
        source.save(self.source)
        result = self.ingest()
        self.assertTrue(result["accepted"])
        repeat = next(w for w in result["warnings"] if w["code"] == "master_repeated_in_activity")
        self.assertEqual(repeat["frames"], [1])
        self.assertEqual(result["structure"], "needs_review")

    def test_absent_mode_preserves_historical_embedded_stand(self):
        manifest_path = self.job / "manifest.json"
        data = sw.read_json(manifest_path)
        del data["sequence_mode"]
        del data["presentation_layout"]
        manifest_path.write_text(json.dumps(data))
        self.set_master()
        self.ingest()
        data = sw.read_json(manifest_path)
        self.assertNotIn("sequence_mode", data)
        attempt = data["tasks"][0]["attempts"][0]
        self.assertEqual((self.job / attempt["frames"][0]["path"]).read_bytes(), self.master.read_bytes())
        self.assertTrue((self.job / attempt["path"] / "generated_slot1_unused.png").is_file())
        self.assertIsNone(attempt["mapping"]["loop_indices"])
        self.assertFalse((self.job / attempt["path"] / "legacy7_atlas.png").exists())
        delivered = self.call(["finalize", "--job", str(self.job)])
        snapshot = sw.read_json(delivered["manifest"])
        self.assertNotIn("presentation_layout", sw.read_json(manifest_path))
        with Image.open(Path(delivered["manifest"]).parent / snapshot["snapshot_atlases"][0]["path"]) as image:
            self.assertEqual(image.size, (576,192))
            with Image.open(self.master) as master:
                self.assertEqual(image.crop((0,0,96,96)).tobytes(), master.tobytes())

    def test_stand_first_finalize_primary_slots_and_separate_six_active_match_pixels(self):
        with Image.open(self.master) as original:
            master = original.copy()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("original_bytes_must_survive", "stand-source-metadata-that-reencoding-would-lose")
        master.save(self.master, pnginfo=metadata)
        self.new_separate_job()
        self.set_master()
        self.ingest()
        result = self.call(["finalize", "--job", str(self.job)])
        snapshot = sw.read_json(result["manifest"])
        atlas = snapshot["snapshot_atlases"][0]
        directory = Path(result["manifest"]).parent
        attempt = snapshot["tasks"][0]["attempts"][0]
        with Image.open(directory / atlas["path"]) as image:
            self.assertEqual(image.size, (672,96))
            with Image.open(self.master) as master:
                self.assertEqual(image.crop((0,0,96,96)).tobytes(), master.tobytes())
            for n, rec in enumerate(attempt["frames"]):
                with Image.open(self.job / rec["path"]) as active:
                    self.assertEqual(image.crop(((n+1)*96,0,(n+2)*96,96)).tobytes(), active.tobytes())
            contact = Image.new("RGBA", image.size, (116,127,123,255))
            contact.alpha_composite(image.convert("RGBA"))
        with Image.open(directory / atlas["contact_native"]["path"]) as image:
            self.assertEqual(image.tobytes(), contact.tobytes(), "Contact uses exactly the primary seven-slot order")
        self.assertEqual(atlas["slot_roles"], ["stand"] + [f"active{n}" for n in range(1,7)])
        self.assertEqual(atlas["loop_indices"], [2,3,4,5,6,7])
        active_atlas = atlas["active_atlas"]
        self.assertEqual(active_atlas["loop_indices"], [1,2,3,4,5,6])
        with Image.open(directory / active_atlas["path"]) as image:
            self.assertEqual(image.size, (576,96))
            for n, rec in enumerate(attempt["frames"]):
                with Image.open(self.job / rec["path"]) as active:
                    self.assertEqual(image.crop((n*96,0,(n+1)*96,96)).tobytes(), active.tobytes())
        presentation = snapshot["snapshot_presentations"][0]
        self.assertEqual(presentation["slot_count"], 7)
        self.assertEqual(presentation["loop_indices"], [2,3,4,5,6,7])
        self.assertEqual(presentation["activity_loop_indices"], [1,2,3,4,5,6])
        first = presentation["slots"][0]
        self.assertEqual(first["slot"], 1)
        self.assertEqual(first["role"], "stand")
        self.assertIsNone(first["active_frame"])
        self.assertEqual((directory / first["path"]).read_bytes(), self.master.read_bytes())
        for n, slot in enumerate(presentation["slots"][1:],start=1):
            self.assertEqual(slot["slot"], n+1)
            self.assertEqual(slot["active_frame"], n)
            self.assertEqual((directory/slot["path"]).read_bytes(), (self.job/attempt["frames"][n-1]["path"]).read_bytes())
        wrapper = atlas["legacy7_wrapper"]
        self.assertEqual(wrapper["loop_indices"], [2,3,4,5,6,7])
        self.assertEqual(wrapper["slot_roles"], ["stand"] + [f"active{n}" for n in range(1,7)])
        with Image.open(directory / wrapper["path"]) as image:
            self.assertEqual(image.size, (672,96))
            for n, rec in enumerate(attempt["frames"]):
                with Image.open(self.job / rec["path"]) as active:
                    self.assertEqual(image.crop(((n+1)*96,0,(n+2)*96,96)).tobytes(), active.tobytes())
        self.assertEqual(result["visual"], "needs_review")
        self.assertEqual(result["user"], "not_reviewed")

    def test_stand_first_partial_direction_rows_keep_seven_slot_primary_and_six_active(self):
        self.job = Path(self.call(["init", "--reference", str(self.master), "--output", str(self.base / "partial-seven"),
                                  "--name", "半成品", "--directions", "right", "left", "--actions", "walk"])["job"])
        self.set_master("left")
        self.ingest("left-walk")
        result = self.call(["finalize", "--job", str(self.job)])
        self.assertEqual(result["delivery_status"], "partial")
        snapshot = sw.read_json(result["manifest"])
        atlas = snapshot["snapshot_atlases"][0]
        directory = Path(result["manifest"]).parent
        with Image.open(directory / atlas["path"]) as image:
            self.assertEqual(image.size, (672,192))
            self.assertIsNone(image.crop((0,0,672,96)).getchannel("A").getbbox())
            with Image.open(self.master) as master:
                self.assertEqual(image.crop((0,96,96,192)).tobytes(), master.tobytes())
        with Image.open(directory / atlas["active_atlas"]["path"]) as image:
            self.assertEqual(image.size, (576,192))
            self.assertIsNone(image.crop((0,0,576,96)).getchannel("A").getbbox())
        self.assertEqual(len(snapshot["snapshot_presentations"]), 1)
        self.assertEqual(snapshot["snapshot_presentations"][0]["direction"], "left")
        self.assertEqual(snapshot["snapshot_presentations"][0]["loop_indices"], [2,3,4,5,6,7])

    def test_stand_first_cannot_wrap_five_active_legacy_storage_as_six(self):
        with self.assertRaisesRegex(sw.WorkflowError, "requires separate_stand"):
            self.call(["init", "--reference", str(self.master), "--output", str(self.base / "invalid-layout"),
                       "--name", "不相容格式", "--sequence-mode", "legacy_embedded_stand", "--presentation-layout", "stand_first7"])

    def test_separate_preview_plays_all_six_without_repeating_stand(self):
        node = shutil.which("node") or ""
        if not Path(node).is_file():
            self.skipTest("Node is required for the executable preview UI regression")
        self.new_separate_job(("walk", "run"))
        self.set_master()
        self.ingest()
        self.ingest("right-run")
        result = self.call(["finalize", "--job", str(self.job)])
        harness = r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const page=fs.readFileSync(process.argv[1],'utf8'),script=page.split('<script>')[1].split('</script>')[0];
const elements=new Map(),raf=[],draws=[];
const defaults={fps:'8',scale:'1',stride:'48',mode:'inplace',loop:'1,2,3,4,5,6',loopMode:'full'};
class Element{constructor(id=''){this.id=id;this.value=defaults[id]||'';this.textContent='';this.children=[];this.style={};this.clientWidth=1000;this.clientHeight=330;}replaceChildren(...children){this.children=children;if(children[0]&&!children.some(c=>c.value===this.value))this.value=children[0].value;}append(...children){this.children.push(...children);}getContext(){return{clearRect(){},drawImage(image,...args){draws.push({source:image.src,args});}};}}
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);},createElement(){return new Element();}};
class MockImage{set src(value){this._src=value;queueMicrotask(()=>this.onload());}get src(){return this._src;}cloneNode(){const image=new MockImage();image._src=this.src;return image;}}
const context=vm.createContext({document,Image:MockImage,window:{addEventListener(){}},requestAnimationFrame(fn){raf.push(fn);}});
vm.runInContext(script,context);
const get=id=>document.getElementById(id),click=id=>get(id).onclick(),flush=()=>new Promise(resolve=>setImmediate(resolve));
const standFirst=process.argv[2]==='stand_first7';
function shown(n){const current=vm.runInContext('current',context);assert.equal(draws.at(-1).source,n===0?current.stand_image:current.images[n-1]);if(standFirst)assert(get('info').textContent.includes('第'+(n+1)+'张 / 7'));else if(n===0)assert(get('info').textContent.includes('STAND / START'));else assert(get('info').textContent.includes(' · F'+n+' / 6'));}
(async()=>{await flush();shown(0);assert.equal(get('frames').children.length,standFirst?7:6);assert.equal(new Set(vm.runInContext('current.images',context)).size,6);
if(standFirst){assert.equal(get('frames').children[0].children[0].src,vm.runInContext('current.stand_image',context));assert.equal(get('loop').value,'2,3,4,5,6,7');get('frames').children[6].onclick();shown(6);get('frames').children[0].onclick();shown(0);}
click('step');shown(1);for(let n=2;n<=6;n++){click('step');shown(n);}click('step');shown(1);
click('stand');shown(0);click('play');shown(1);let now=1000;raf.shift()(now);for(let n=2;n<=6;n++){raf.shift()(now+=62.5);raf.shift()(now+=62.5);shown(n);}raf.shift()(now+=62.5);raf.shift()(now+=62.5);shown(1);
click('play');click('stand');shown(0);get('loop').value=standFirst?'3,5,7':'2,4,6';click('apply');shown(0);assert(get('phase').textContent.includes('临时观察实验'));assert(get('phase').textContent.includes('完整六帧'));assert.equal(vm.runInContext('JSON.stringify(current.loop_indices)',context),'[1,2,3,4,5,6]');for(const n of[2,4,6,2]){click('step');shown(n);}
if(standFirst){const before=vm.runInContext('JSON.stringify(loop)',context);get('loop').value='1,2,3,4,5,6,7';click('apply');assert.equal(vm.runInContext('JSON.stringify(loop)',context),before,'stand cannot enter repeating activity');}
get('loopMode').value='full';get('loopMode').onchange();shown(0);for(const n of[1,2,3,4,5,6,1]){click('step');shown(n);}
function moving(scale,startTime){
  click('stand');get('scale').value=String(scale);get('scale').onchange();get('mode').value='world';get('mode').onchange();click('play');shown(1);
  const screenStart=draws.at(-1).args[0];let time=startTime;raf.shift()(time);
  for(let n=0;n<12;n++)raf.shift()(time+=62.5);
  shown(1);const screenDelta=draws.at(-1).args[0]-screenStart;
  assert.equal(vm.runInContext('worldX',context),48,'one 6-frame cycle moves one native 48px stride');
  assert.equal(screenDelta/scale,48,'rendered native displacement is independent of observation scale');
  click('play');return screenDelta;
}
assert.equal(moving(3,5000),3*moving(1,7000),'3x observation scales rendered displacement by exactly three');
get('scale').value='3';get('scale').onchange();get('stage').width=320;get('stage').height=280;
vm.runInContext('worldX=110;worldY=100',context);click('play');raf.shift()(9000);
assert.equal(vm.runInContext('worldX',context),-110,'horizontal wrap uses native viewport width / scale');
assert.equal(vm.runInContext('worldY',context),-100,'vertical wrap uses native viewport height / scale');
get('action').value='run';await get('action').onchange();shown(0);assert.equal(vm.runInContext('current.task',context),'right-run');click('play');shown(1);if(standFirst){assert(get('info').textContent.includes('跑步1/6'));assert(get('frames').children[1].children[1].textContent.includes('跑步1'));assert(get('phase').textContent.includes('跑步循环'));}
assert(get('status').textContent.includes('用户未认可'));console.log('ACTIVE6_PREVIEW_EXECUTION_PASS: separate stand; all six active; wrap 6-to-1; legacy export independent');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
        for layout in ("stand_first7", "old_without_layout"):
            if layout == "old_without_layout":
                data = sw.read_json(self.job / "manifest.json")
                del data["presentation_layout"]
                (self.job / "manifest.json").write_text(json.dumps(data))
                result = self.call(["finalize", "--job", str(self.job)])
                self.assertNotIn("presentation_layout", sw.read_json(self.job / "manifest.json"))
                legacy = sw.read_json(result["manifest"])["snapshot_atlases"][0]
                with Image.open(Path(result["manifest"]).parent / legacy["path"]) as image:
                    self.assertEqual(image.size, (576,96), "Pre-layout separate_stand keeps six-slot primary")
            executed = subprocess.run([node, "-e", harness, result["preview"], layout], capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertIn("ACTIVE6_PREVIEW_EXECUTION_PASS", executed.stdout)
        snapshot = sw.read_json(result["manifest"])
        self.assertEqual(snapshot["tasks"][0]["attempts"][0]["mapping"]["loop_indices"], [1,2,3,4,5,6])
        self.assertEqual(snapshot["review"]["visual"], "needs_review")


if __name__ == "__main__":
    unittest.main(verbosity=2)
