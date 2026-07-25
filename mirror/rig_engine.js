// rig_engine.js — reusable procedural animation engine for the Bluey/Sanrio GLB rigs.
//
// Works with any model from this collection (shared Blender skeleton: Hips/Spine.001/
// Spine.002/Head, UpperArm/forearm/Hand .L/.R, thigh/shin/Foot/Toe .L/.R, Tail chain,
// Ear chains, optional fingers and accessory chains like Bow/Dress/Hat).
//
// Usage in a game:
//   import { RigCharacter } from './rig_engine.js';
//   const gltf = await new GLTFLoader().loadAsync('Bluey1_fixed.glb');
//   const rig = new RigCharacter(gltf.scene, THREE);   // fixes materials, maps skeleton
//   scene.add(gltf.scene);
//   rig.play('walk', { speed: 1.2 });                  // built-in clip …
//   rig.pose('Head', 0, yaw, 0);                       // … or drive bones directly
//   // each frame:
//   rig.update(dt);
//
// pose() takes a WORLD-space euler delta (radians) applied on top of the bone's rest
// pose — you never need to know a bone's local axes. Bone names are the glTF names
// with punctuation stripped, exactly as three.js imports them: "UpperArm.L" -> "UpperArmL".

export class RigCharacter {
    constructor(root, THREE, opts = {}) {
        this.THREE = THREE;
        this.root = root;
        this.smoothing = opts.smoothing ?? 14;     // higher = snappier
        this.relaxAfter = opts.relaxAfter ?? 0.5;  // s without input before easing to rest
        this.secondaryMotion = opts.secondaryMotion ?? true;
        this.lidSign = opts.lidSign ?? 1;          // flip if bone-based blink opens instead of closes

        this.bones = {};        // name -> {bone, restLocal, restWorld, restWorldInv, target, lastSet}
        this.morphMeshes = [];
        this.morphs = {};       // name -> {value, target, lastSet, rest}
        this._clip = null;
        this._clipOpts = {};
        this._t = 0;
        this._springs = [];

        this._e = new THREE.Euler();
        this._q = new THREE.Quaternion();
        this._q2 = new THREE.Quaternion();

        if (opts.fixMaterials !== false) RigCharacter.fixMaterials(root, THREE);

        root.updateMatrixWorld(true);
        root.traverse((c) => {
            if (c.isMesh && c.morphTargetDictionary) {
                this.morphMeshes.push(c);
                for (const key of Object.keys(c.morphTargetDictionary))
                    if (!this.morphs[key]) this.morphs[key] = { value: 0, target: 0, lastSet: -1, rest: 0 };
            }
            if (c.isBone) {
                const rw = new THREE.Quaternion();
                c.getWorldQuaternion(rw);
                this.bones[c.name] = {
                    bone: c, restLocal: c.quaternion.clone(), restPosLocal: c.position.clone(),
                    restWorld: rw.clone(), restWorldInv: rw.clone().invert(),
                    target: c.quaternion.clone(), lastSet: -1
                };
            }
        });
        this.semantic = this._mapSemantic();
        this._buildSprings();
        this._cal = null;
        this.collision = opts.collision !== false ? new CollisionGuard(this) : null;
        this.armDownAuto = this._measureArmDown();
        // Phase-4 locomotion infrastructure. Assumes the root was normalized
        // (RigCharacter.normalize) BEFORE construction, as in all pipeline flows.
        this.baseRootPos = root.position.clone(); // jump returns here instead of stomping y=0
        this.legLengthAuto = this._measureLegLength(); // world units, thigh→shin→foot
        this._hipsDriver = this._buildHipsDriver();
        // Phase-5: measure FACING geometrically (+1 = faces world +z) BEFORE the
        // leg-axis probes — knee flexion is defined relative to facing, and the
        // walk gait direction is oriented by it. All 12 production models face
        // +z; a Phase-4 assumption of −z produced a backwards (moonwalk) gait.
        this.facing = this._measureFacing();
        this.legAxes = this._measureLegAxes(); // per-bone local→flexion axis map (defect E)
        // Empirical ankle-height-vs-thigh-pitch curve (rest legs are splayed,
        // so analytic legLen·cos θ misestimates reach by up to ~15 mm).
        this._legReach = this._measureLegReach();
        // Phase-5: rest arm directions for the fore-aft swing axis (walk arms).
        this._armRestDirL = this._measureArmDir('L');
        this._armRestDirR = this._measureArmDir('R');
        // scratch objects for _poseArmSwing (no per-frame allocation)
        this._armTmpDir = new THREE.Vector3();
        this._armTmpAxis = new THREE.Vector3();
        this._armTmpFwd = new THREE.Vector3();
        this._armTmpQ = new THREE.Quaternion();
        this._armTmpQ2 = new THREE.Quaternion();
    }

    // Per-model tuning overrides, e.g. { clips: { walk: { armAmp: 0.7, armRollOut: 0.3 } } }.
    // Produced by rig_audit.html; merged into clip opts on the next play().
    calibrate(cal) { this._cal = cal; }

    // --- static helpers -------------------------------------------------

    // These GLBs export every material as alpha-BLEND which breaks depth sorting
    // (see-through faces). Force opaque with an alpha cutout.
    static fixMaterials(root, THREE) {
        root.traverse((c) => {
            if (!c.isMesh) return;
            const mats = Array.isArray(c.material) ? c.material : [c.material];
            const allHaveTexture = mats.every(m => m.map);
            for (const m of mats) {
                m.side = THREE.DoubleSide;
                m.transparent = false;
                m.depthWrite = true;
                m.alphaTest = 0;
                if (m.emissive) m.emissive.setRGB(0, 0, 0);
                m.roughness = 0.92;
                if (allHaveTexture) m.vertexColors = false;
                m.needsUpdate = true;
            }
            if (allHaveTexture && c.geometry?.attributes?.color) c.geometry.deleteAttribute('color');
            c.frustumCulled = false;
        });
    }

    // Bone-aware world bounding box. Box3.setFromObject uses raw bind-pose geometry,
    // which on these rigs can be a different size than what the bones render —
    // SkinnedMesh.computeBoundingBox (r151+) applies the actual bone transforms.
    static bounds(root, THREE) {
        root.updateMatrixWorld(true);
        let box = null;
        root.traverse((c) => {
            if (!c.isMesh) return;
            let b;
            if (c.isSkinnedMesh && c.computeBoundingBox) { c.computeBoundingBox(); b = c.boundingBox.clone(); }
            else {
                if (!c.geometry.boundingBox) c.geometry.computeBoundingBox();
                b = c.geometry.boundingBox.clone();
            }
            b.applyMatrix4(c.matrixWorld);
            box = box ? box.union(b) : b;
        });
        return box ?? new THREE.Box3(new THREE.Vector3(), new THREE.Vector3());
    }

    // Uniform-scale the model to `height` units tall with feet on y=0. Idempotent —
    // safe to call again on an already-normalized model (multiplies, not sets).
    static normalize(root, THREE, height = 2.2) {
        const box = RigCharacter.bounds(root, THREE);
        const size = box.getSize(new THREE.Vector3());
        const s = height / (size.y || 1);
        root.scale.multiplyScalar(s);
        const box2 = RigCharacter.bounds(root, THREE);
        root.position.y -= box2.min.y;
        return s;
    }

    // --- skeleton discovery ---------------------------------------------

    _chain(prefix) {
        // Match both dotted (Blender default: "Ear.L") and undotted ("EarL") names,
        // plus numbered suffixes like "Spine.001" or "Spine001"
        return Object.keys(this.bones)
            .filter(n => {
                if (n === prefix) return true;
                // Strip dots for comparison: "forearm.L" → "forearmL", "Spine.001" → "Spine001"
                const stripped = n.replace(/\./g, '');
                if (stripped === prefix) return true;
                // Check numbered suffix: "Tail.001" matches prefix "Tail", suffix "001"
                if (n.startsWith(prefix) && /^\.?\d+$/.test(n.slice(prefix.length))) return true;
                if (stripped.startsWith(prefix) && /^\d+$/.test(stripped.slice(prefix.length))) return true;
                return false;
            })
            .sort();
    }
    _mapSemantic() {
        // Try exact name first, then with dot before L/R suffix (Blender convention)
        const has = n => {
            if (this.bones[n]) return n;
            // "forearmL" → try "forearm.L"; "HandR" → try "Hand.R"
            const dotted = n.replace(/([LR])$/, '.$1');
            if (dotted !== n && this.bones[dotted]) return dotted;
            return null;
        };
        const s = {
            hips: has('Hips'), head: has('Head'), snout: has('Snout'),
            spine: this._chain('Spine'),
            upperArmL: has('UpperArmL'), forearmL: has('forearmL'), handL: has('HandL'), shoulderL: has('shoulderL'),
            upperArmR: has('UpperArmR'), forearmR: has('forearmR'), handR: has('HandR'), shoulderR: has('shoulderR'),
            thighL: has('thighL'), shinL: has('shinL'), footL: has('FootL'), toeL: has('ToeL'),
            thighR: has('thighR'), shinR: has('shinR'), footR: has('FootR'), toeR: has('ToeR'),
            tail: this._chain('Tail'),
            earL: this._chain('EarL') .length ? this._chain('EarL')  : this._chain('Ear.L'),
            earR: this._chain('EarR') .length ? this._chain('EarR')  : this._chain('Ear.R'),
            lidTopL: this._chain('EyeLidTopL') .length ? this._chain('EyeLidTopL')  : this._chain('EyeLidTop.L'),
            lidTopR: this._chain('EyeLidTopR') .length ? this._chain('EyeLidTopR')  : this._chain('EyeLidTop.R'),
            lidBotL: this._chain('EyeLidBottomL') .length ? this._chain('EyeLidBottomL')  : this._chain('EyeLidBottom.L'),
            lidBotR: this._chain('EyeLidBottomR') .length ? this._chain('EyeLidBottomR')  : this._chain('EyeLidBottom.R'),
            fingersL: {}, fingersR: {},
            dangly: []
        };
        for (const f of ['Index', 'Mid', 'Pinky', 'Thumb']) {
            let l = this._chain(f + 'L'), r = this._chain(f + 'R');
            if (!l.length) l = this._chain(f + '.L');
            if (!r.length) r = this._chain(f + '.R');
            if (l.length) s.fingersL[f] = l;
            if (r.length) s.fingersR[f] = r;
        }
        for (const prefix of ['Bow', 'Hat', 'DressL', 'DressR', 'DressB', 'DressF',
                               'Dress.L', 'Dress.R', 'Dress.B', 'Dress.F',
                               // kuromi V2 jester collar: two 9-bone chains that
                               // otherwise stay rigid (Phase-5 defect fix)
                               'jester_collarL', 'jester_collarR',
                               'jester_collar.L', 'jester_collar.R']) {
            const c = this._chain(prefix);
            if (c.length) s.dangly.push(c);
        }
        return s;
    }

    // Rest-pose world direction of an arm chain (upperArm → hand, or forearm
    // tip if no hand). Used to build the fore-aft swing axis in walk: swinging
    // about raw world X is half TWIST for rolled-out chibi arms (armDown≈0.5),
    // which made kuromi V2 / Sanrio hands read as frozen (87 mm walk hand
    // travel vs Bluey 518 mm).
    _measureArmDir(side) {
        const s = this.semantic, THREE = this.THREE;
        const up = this.bones[s['upperArm' + side]]?.bone;
        const end = this.bones[s['hand' + side]]?.bone
                 || this.bones[s['forearm' + side]]?.bone;
        if (!up || !end) return new THREE.Vector3(0, -1, 0);
        const a = up.getWorldPosition(new THREE.Vector3());
        const b = end.getWorldPosition(new THREE.Vector3());
        const d = b.sub(a);
        return d.lengthSq() > 1e-10 ? d.normalize() : new THREE.Vector3(0, -1, 0);
    }

    // Measure how horizontal the upper arm is in rest pose.
    // Returns a good armDown value: ~1.4 for cartoon T-pose arms, ~1.1 for humanoids.
    _measureArmDown() {
        const bone = this.bones[this.semantic.upperArmL]?.bone
                  || this.bones[this.semantic.upperArmR]?.bone;
        if (!bone) return 1.4;
        const dir = new this.THREE.Vector3();
        bone.getWorldDirection(dir);
        // How horizontal is the arm? dot with world X axis (arms point sideways in T-pose)
        const horizontalness = Math.abs(dir.x);
        // horizontalness ~1 = fully horizontal (cartoon) → needs more armDown
        // horizontalness ~0 = pointing forward/down (humanoid) → needs less
        return 1.1 + horizontalness * 0.35; // range: 1.1 (humanoid) → 1.45 (cartoon)
    }

    // Leg chain length in world units (thigh→shin + shin→foot, rest pose).
    // Drives auto stride/bob scaling so the same clip math works on Bluey-length
    // legs (~0.5u) and Sanrio stub legs (~0.2u). Also records _stanceSign, the
    // world-X sign of the LEFT leg, used to shift weight toward the stance foot.
    _measureLegLength() {
        const s = this.semantic, THREE = this.THREE;
        for (const side of ['L', 'R']) {
            const th = this.bones[s['thigh' + side]]?.bone,
                  sh = this.bones[s['shin' + side]]?.bone,
                  ft = this.bones[s['foot' + side]]?.bone;
            if (!th || !sh || !ft) continue;
            const a = th.getWorldPosition(new THREE.Vector3()),
                  b = sh.getWorldPosition(new THREE.Vector3()),
                  c = ft.getWorldPosition(new THREE.Vector3());
            if (side === 'L') this._stanceSign = Math.sign(a.x) || 1;
            return a.distanceTo(b) + b.distanceTo(c);
        }
        this._stanceSign = this._stanceSign ?? 1;
        return 0.5;
    }

    // Hips translation driver: world +Y/+X/+Z expressed in the hips parent's
    // local space, so clips can translate the pelvis (bob, weight shift,
    // crouch, anti-slide fore-aft) in world units without knowing the rig's
    // bone axes or the root scale.
    _buildHipsDriver() {
        const THREE = this.THREE;
        const entry = this.bones[this.semantic.hips];
        const hips = entry?.bone;
        if (!hips || !hips.parent) return null;
        const pq = hips.parent.getWorldQuaternion(new THREE.Quaternion()).invert();
        const ps = hips.parent.getWorldScale(new THREE.Vector3());
        const dirY = new THREE.Vector3(0, 1, 0).applyQuaternion(pq);
        const dirX = new THREE.Vector3(1, 0, 0).applyQuaternion(pq);
        const dirZ = new THREE.Vector3(0, 0, 1).applyQuaternion(pq);
        for (const d of [dirY, dirX, dirZ]) { d.x /= ps.x; d.y /= ps.y; d.z /= ps.z; }
        return { bone: hips, restPos: entry.restPosLocal, dirY, dirX, dirZ };
    }

    // Translate the pelvis by (dy up, dx toward +X, dz toward +Z) in WORLD units
    // from rest. Unlike pose() this writes position directly (no smoothing) —
    // intended to be re-applied every frame by the active clip. reset()
    // restores restPos.
    _driveHips(dy = 0, dx = 0, dz = 0) {
        const d = this._hipsDriver; if (!d) return false;
        d.bone.position.copy(d.restPos)
            .addScaledVector(d.dirY, dy)
            .addScaledVector(d.dirX, dx)
            .addScaledVector(d.dirZ, dz);
        return true;
    }

    // Phase-5: geometric facing detection — majority vote over independent
    // signals (toe bone vs foot bone, toe-vs-foot VERTEX centroids, tail base
    // behind hips, snout ahead of head). Returns ±1: the world-Z sign the
    // character faces. Defaults to +1 when no signal fires (every production
    // model measures +1 with high agreement; see audit/facing_probe.mjs).
    _measureFacing() {
        const s = this.semantic, THREE = this.THREE;
        const W = (n) => this.bones[n]?.bone.getWorldPosition(new THREE.Vector3());
        const sigs = [];
        const f = W(s.footL) ?? W(s.footR), t = W(s.toeL) ?? W(s.toeR);
        if (f && t && Math.abs(t.z - f.z) > 0.003) sigs.push(Math.sign(t.z - f.z));
        const h = W(s.hips), t0 = s.tail?.length ? W(s.tail[0]) : null;
        if (h && t0 && Math.abs(t0.z - h.z) > 0.003) sigs.push(-Math.sign(t0.z - h.z));
        const hd = W(s.head), sn = W(s.snout);
        if (hd && sn && Math.abs(sn.z - hd.z) > 0.003) sigs.push(Math.sign(sn.z - hd.z));
        // vertex-level: mean z of toe-dominant verts minus foot-dominant verts
        let zToe = 0, nToe = 0, zFoot = 0, nFoot = 0;
        const v = new THREE.Vector3();
        const si_sw = (mesh, i) => {
            const si = mesh.geometry.attributes.skinIndex, sw = mesh.geometry.attributes.skinWeight;
            let bi = si.getX(i), bw = sw.getX(i);
            if (sw.getY(i) > bw) { bw = sw.getY(i); bi = si.getY(i); }
            if (sw.getZ(i) > bw) { bw = sw.getZ(i); bi = si.getZ(i); }
            if (sw.getW(i) > bw) { bw = sw.getW(i); bi = si.getW(i); }
            return mesh.skeleton.bones[bi]?.name ?? '';
        };
        this.root.traverse((mesh) => {
            if (!mesh.isSkinnedMesh) return;
            const pos = mesh.geometry.attributes.position;
            if (!pos || !mesh.geometry.attributes.skinIndex) return;
            const step = Math.max(1, Math.floor(pos.count / 4000));
            for (let i = 0; i < pos.count; i += step) {
                const norm = si_sw(mesh, i).toLowerCase().replace(/[^a-z0-9]/g, '');
                mesh.getVertexPosition(i, v); v.applyMatrix4(mesh.matrixWorld);
                if (/^toe/.test(norm)) { zToe += v.z; nToe++; }
                else if (/^foot/.test(norm)) { zFoot += v.z; nFoot++; }
            }
        });
        if (nToe > 5 && nFoot > 5) {
            const dz = zToe / nToe - zFoot / nFoot;
            if (Math.abs(dz) > 0.003) sigs.push(Math.sign(dz));
        }
        const score = sigs.reduce((a, b) => a + b, 0);
        return score === 0 ? 1 : Math.sign(score);
    }

    // Phase-4 defect E: on these Blender rigs a shin/foot bone's local X is a
    // sideways ROLL axis, not the flexion axis — poseLocal(shin, x) scissored
    // the lower leg sideways instead of bending the knee, and the .L/.R chains
    // are not consistently mirrored across models. So measure, per bone, which
    // local axis + sign produces anatomical flexion, by probing the actual
    // skeleton at construction:
    //   knee: +1 command = flexion (foot tip moves AWAY from the facing dir)
    //   foot: +1 command = world +x pitch (dorsiflex; used by the sole-flatten
    //         compensation, which works in world-pitch units)
    // Also records fw: the world-Z sign a foot moves when the thigh is pitched
    // +x (i.e. the character's facing direction), needed by jump's toe-point.
    _measureLegAxes() {
        const s = this.semantic, THREE = this.THREE;
        const out = { fw: -1 };
        const probe = (boneName, tipName, scoreFn) => {
            const entry = this.bones[boneName], tip = this.bones[tipName];
            if (!entry || !tip) return null;
            const bone = entry.bone;
            const q0 = bone.quaternion.clone();
            const tip0 = tip.bone.getWorldPosition(new THREE.Vector3());
            let best = null;
            for (const axis of ['x', 'z']) {           // local Y runs along the bone
                for (const sign of [1, -1]) {
                    const e = new THREE.Euler(
                        axis === 'x' ? 0.2 * sign : 0, 0,
                        axis === 'z' ? 0.2 * sign : 0, 'XYZ');
                    bone.quaternion.copy(q0).multiply(new THREE.Quaternion().setFromEuler(e));
                    this.root.updateMatrixWorld(true);
                    const d = tip.bone.getWorldPosition(new THREE.Vector3()).sub(tip0);
                    const score = scoreFn(d);
                    if (!best || score > best.score) best = { axis, sign, score };
                    bone.quaternion.copy(q0);
                }
            }
            this.root.updateMatrixWorld(true);
            return best ? { axis: best.axis, sign: best.sign } : null;
        };
        // facing direction: pitch thighL +x world, watch where the foot goes
        const th = this.bones[s.thighL], ft = this.bones[s.footL];
        if (th && ft) {
            const e = new THREE.Euler(0.25, 0, 0, 'XYZ');
            const dq = new THREE.Quaternion().setFromEuler(e);
            const local = th.restWorldInv.clone().multiply(dq).multiply(th.restWorld);
            const q0 = th.bone.quaternion.clone();
            const f0 = ft.bone.getWorldPosition(new THREE.Vector3());
            th.bone.quaternion.copy(q0).multiply(local);
            this.root.updateMatrixWorld(true);
            const dz = ft.bone.getWorldPosition(new THREE.Vector3()).z - f0.z;
            th.bone.quaternion.copy(q0);
            this.root.updateMatrixWorld(true);
            if (Math.abs(dz) > 1e-5) out.fw = Math.sign(dz);
        }
        // knee flexion: foot tip moves opposite to FACING (backward). Phase-4
        // keyed this to out.fw (the thigh-pitch axis, −1 on all models) which
        // stands in for facing only when the character faces −z; on these +z
        // facing rigs it selected forward-bending ("bird") knees.
        const F = this.facing ?? 1;
        const kneeScore = (d) => (-F * d.z) - 0.3 * Math.abs(d.x);
        // foot dorsiflex (+world x pitch): toe tip dips in z (heel-raise sense),
        // penalize sideways
        const footScore = (d) => (-d.z) - 0.3 * Math.abs(d.x);
        for (const side of ['L', 'R']) {
            out['knee' + side] = probe(s['shin' + side], s['foot' + side], kneeScore);
            out['foot' + side] = probe(s['foot' + side], s['toe' + side] ?? s['foot' + side], footScore);
        }
        return out;
    }

    // Phase-5: empirically sample ankle world height vs thigh world-x pitch,
    // with the walk's stance knee (KNEE_MIN) and sole-flatten ankle applied.
    // The rest legs on these chibi rigs are splayed (Bluey: 239 mm chain,
    // 161 mm vertical drop), so the analytic legLen·(1−cos θ) reach model
    // misestimates the support height by up to ~15 mm and cannot express the
    // front/back reach asymmetry of a rest pose that isn't fore-aft centered.
    // Returns { L: {max, y[], y0}, R: ... } sampled over θ ∈ ±max, or null.
    _measureLegReach() {
        const s = this.semantic, THREE = this.THREE;
        const MAX = 0.8, N = 33, KNEE = 0.12;
        const out = {};
        for (const side of ['L', 'R']) {
            const th = this.bones[s['thigh' + side]], sh = this.bones[s['shin' + side]], ft = this.bones[s['foot' + side]];
            if (!th || !ft) continue;
            const km = this.legAxes?.['knee' + side], fm = this.legAxes?.['foot' + side];
            const qT = th.bone.quaternion.clone(), qS = sh?.bone.quaternion.clone(), qF = ft.bone.quaternion.clone();
            const y = [];
            for (let i = 0; i < N; i++) {
                const pitch = -MAX + (2 * MAX * i) / (N - 1);
                const dq = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, 0, 0, 'XYZ'));
                th.bone.quaternion.copy(qT).multiply(th.restWorldInv.clone().multiply(dq).multiply(th.restWorld));
                if (sh && km) sh.bone.quaternion.copy(qS).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(
                    km.axis === 'x' ? KNEE * km.sign : 0, km.axis === 'y' ? KNEE * km.sign : 0, km.axis === 'z' ? KNEE * km.sign : 0, 'XYZ')));
                if (fm) {
                    const v = -(pitch + KNEE);
                    ft.bone.quaternion.copy(qF).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(
                        fm.axis === 'x' ? v * fm.sign : 0, fm.axis === 'y' ? v * fm.sign : 0, fm.axis === 'z' ? v * fm.sign : 0, 'XYZ')));
                }
                this.root.updateMatrixWorld(true);
                y.push(ft.bone.getWorldPosition(new THREE.Vector3()).y);
            }
            th.bone.quaternion.copy(qT); if (sh) sh.bone.quaternion.copy(qS); ft.bone.quaternion.copy(qF);
            this.root.updateMatrixWorld(true);
            const toe = this.bones[s['toe' + side]]?.bone;
            const footLen = toe ? ft.bone.getWorldPosition(new THREE.Vector3())
                .distanceTo(toe.getWorldPosition(new THREE.Vector3())) : 0.1;
            out[side] = { max: MAX, y, y0: y[(N - 1) / 2], footLen };
        }
        return Object.keys(out).length ? out : null;
    }

    // Linear-interpolated ankle height at thigh pitch x from the sampled curve.
    _legReachAt(side, x) {
        const c = this._legReach?.[side];
        if (!c) return null;
        const n = c.y.length;
        const u = Math.min(n - 1.0001, Math.max(0, (x / c.max * 0.5 + 0.5) * (n - 1)));
        const i = Math.floor(u), f = u - i;
        return c.y[i] * (1 - f) + c.y[i + 1] * f;
    }

    // poseLocal through a measured axis map: applies v*map.sign about map.axis.
    _poseMapped(name, map, v) {
        if (!map) return false;
        const a = map.axis === 'x' ? v * map.sign : 0;
        const b = map.axis === 'y' ? v * map.sign : 0;
        const c = map.axis === 'z' ? v * map.sign : 0;
        return this.poseLocal(name, a, b, c);
    }

    _buildSprings() {
        this._springs = [];
        const add = (chain, axis, amp) => chain?.length && this._springs.push({ chain, axis, amp, p: 0, v: 0 });
        add(this.semantic.earL, 'z', 0.5); add(this.semantic.earR, 'z', -0.5);
        for (const c of this.semantic.dangly) add(c, 'x', 0.4);
    }

    // --- posing API ------------------------------------------------------

    // World-space euler delta on top of rest pose. localNew = restLocal * (restWorldInv * D * restWorld)
    pose(name, x = 0, y = 0, z = 0) {
        const b = this.bones[name]; if (!b) return false;
        this._e.set(x, y, z, 'XYZ');
        this._q.setFromEuler(this._e);
        this._q2.copy(b.restWorldInv).multiply(this._q).multiply(b.restWorld);
        b.target.copy(b.restLocal).multiply(this._q2);
        b.lastSet = this._t;
        return true;
    }
    // World-space QUATERNION delta on top of rest pose — same composition as
    // pose() but takes the delta directly (axis-angle swings, e.g. walk arms).
    poseQuat(name, q) {
        const b = this.bones[name]; if (!b) return false;
        this._q2.copy(b.restWorldInv).multiply(q).multiply(b.restWorld);
        b.target.copy(b.restLocal).multiply(this._q2);
        b.lastSet = this._t;
        return true;
    }
    // Walk/run arm driver: hang the arm by downZ (world Z, Phase-4 armDown
    // semantics), then swing it about the axis that moves the hand PURELY
    // fore-aft — cross(posedArmDir, facing) — instead of raw world X, which is
    // mostly twist once arms are rolled out. Sign convention matches the old
    // world-X euler exactly for vertical arms, so Bluey-family gait is
    // unchanged; chibi rolled-out arms gain up to 2× hand travel per armAmp.
    _poseArmSwing(name, side, downZ, swing) {
        if (!name || !this.bones[name]) return false;
        const rest = side === 'L' ? this._armRestDirL : this._armRestDirR;
        const fwd = this._armTmpFwd.set(0, 0, this.facing ?? 1);
        this._armTmpQ.setFromAxisAngle(this._armTmpAxis.set(0, 0, 1), downZ);
        const dir = this._armTmpDir.copy(rest).applyQuaternion(this._armTmpQ);
        const axis = this._armTmpAxis.crossVectors(dir, fwd);
        if (axis.lengthSq() < 1e-6) axis.set(-(this.facing ?? 1), 0, 0); else axis.normalize();
        this._armTmpQ2.setFromAxisAngle(axis, swing).multiply(this._armTmpQ); // down first, then swing
        return this.poseQuat(name, this._armTmpQ2);
    }
    // Local-space euler delta (for bones with meaningful local axes, e.g. eyelids).
    poseLocal(name, x = 0, y = 0, z = 0) {
        const b = this.bones[name]; if (!b) return false;
        this._e.set(x, y, z, 'XYZ');
        this._q.setFromEuler(this._e);
        b.target.copy(b.restLocal).multiply(this._q);
        b.lastSet = this._t;
        return true;
    }
    morph(name, v) {
        const m = this.morphs[name]; if (!m) return false;
        m.target = Math.min(1, Math.max(0, v));
        m.lastSet = this._t;
        return true;
    }
    // Distribute a world-euler along a bone chain (tails, ears, spines).
    poseChain(chain, x = 0, y = 0, z = 0, falloff = 1) {
        if (!chain?.length) return;
        let w = 1, total = 0;
        const ws = chain.map(() => { const v = w; total += w; w *= falloff; return v; });
        chain.forEach((n, i) => this.pose(n, x * ws[i] / total * chain.length / chain.length, y * ws[i] / total, z * ws[i] / total));
    }

    // 0 = open, 1 = closed. Uses Blink morphs when the model has them, else eyelid bones.
    blink(l, r = l) {
        if (this.morphs['Blink']) { this.morph('Blink', Math.max(l, r)); return; }
        if (this.morphs['BlinkL'] || this.morphs['BlinkR']) { this.morph('BlinkL', l); this.morph('BlinkR', r); return; }
        const dist = [0.62, 0.28, 0.10];
        const sweep = (chain, amt, max) =>
            chain.forEach((n, i) => this.poseLocal(n, this.lidSign * max * amt * (dist[i] ?? 0), 0, 0));
        sweep(this.semantic.lidTopL, l, 1.05); sweep(this.semantic.lidTopR, r, 1.05);
        sweep(this.semantic.lidBotL, l, -0.35); sweep(this.semantic.lidBotR, r, -0.35);
    }

    // Viseme by vowel letter, mapped to whichever morph naming the model uses.
    viseme(vowel, amt) {
        const table = {
            A: ['Mouth A', 'A'], E: ['Mouth E', 'E'], I: ['Mouth i', 'Mouth I', 'I'],
            O: ['Mouth O', 'O'], U: ['Mouth U', 'U'], N: ['Mouth N', 'N']
        };
        for (const name of table[vowel] ?? []) if (this.morph(name, amt)) return true;
        return false;
    }

    listBones() { return Object.keys(this.bones); }
    listMorphs() { return Object.keys(this.morphs); }

    reset() {
        for (const n in this.bones) {
            const b = this.bones[n];
            b.target.copy(b.restLocal);
            b.bone.quaternion.copy(b.restLocal);
            b.bone.position.copy(b.restPosLocal); // undo _driveHips translations
            b.lastSet = -1;
        }
        if (this.baseRootPos) this.root.position.copy(this.baseRootPos); // undo jump lift
        for (const n in this.morphs) { const m = this.morphs[n]; m.target = m.rest; m.value = m.rest; m.lastSet = -1; }
        this._applyMorphs();
        this._clip = null;
    }

    // --- clips ------------------------------------------------------------

    play(name, opts = {}) {
        if (!CLIPS[name]) throw new Error(`Unknown clip "${name}". Available: ${Object.keys(CLIPS).join(', ')}`);
        this._clip = name;
        this._clipOpts = { ...(this._cal?.clips?.[name] ?? {}), ...opts };
        this._clipStart = this._t;
    }
    stop() { this._clip = null; }
    get playing() { return this._clip; }

    // --- per-frame update -------------------------------------------------

    update(dt) {
        dt = Math.min(dt, 0.05);
        this._t += dt;
        const t = this._t;

        if (this._clip) CLIPS[this._clip](this, t - this._clipStart, this._clipOpts);
        if (this.secondaryMotion) this._updateSprings(t, dt);
        if (this.collision?.enabled) this.collision.update(dt);

        const kBone = 1 - Math.exp(-this.smoothing * dt);
        const kMorph = 1 - Math.exp(-(this.smoothing + 6) * dt);
        for (const n in this.bones) {
            const b = this.bones[n];
            if (b.lastSet >= 0 && t - b.lastSet > this.relaxAfter)
                b.target.slerp(b.restLocal, 1 - Math.exp(-3 * dt));
            if (b.lastSet === t) {
                b.bone.quaternion.copy(b.target);
            } else {
                b.bone.quaternion.slerp(b.target, kBone);
            }
        }
        for (const n in this.morphs) {
            const m = this.morphs[n];
            if (m.lastSet >= 0 && t - m.lastSet > this.relaxAfter) m.target = m.rest;
            m.value += (m.target - m.value) * kMorph;
        }
        this._applyMorphs();
    }

    _applyMorphs() {
        for (const mesh of this.morphMeshes) {
            for (const n in this.morphs) {
                const idx = mesh.morphTargetDictionary[n];
                if (idx !== undefined) mesh.morphTargetInfluences[idx] = this.morphs[n].value;
            }
        }
    }

    _updateSprings(t, dt) {
        // idle tail wag (only when no clip is driving the tail explicitly)
        if (this.semantic.tail.length && !this._clip)
            this.semantic.tail.forEach((n, i) =>
                this.pose(n, 0, Math.sin(t * 3.4 - i * 0.7) * (0.10 + i * 0.08), 0));
        // dangly chains: light lag springs with an idle breeze
        for (const s of this._springs) {
            const drive = Math.sin(t * 1.3 + s.amp * 7) * 0.02;
            const k = 50, d = 7;
            s.v += (-k * (s.p - drive) - d * s.v) * dt;
            s.p += s.v * dt;
            s.chain.forEach((n, i) => {
                const e = { x: 0, y: 0, z: 0 };
                e[s.axis === 'z' ? 'z' : 'x'] = s.p * s.amp * (1 + i * 0.6);
                this.pose(n, e.x, e.y, e.z);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in procedural clips. Each is (rig, t, opts) -> sets targets via rig API.
// They only touch parts the model actually has, so they run on every rig.
// ---------------------------------------------------------------------------
export const CLIPS = {
    idle(rig, t, { energy = 1, armDown = rig.armDownAuto ?? 1.4 } = {}) {
        const s = rig.semantic;
        rig._driveHips(0, 0); // re-center pelvis if a locomotion clip ran before
        if (s.spine[0]) rig.pose(s.spine[0], Math.sin(t * 1.5) * 0.02 * energy, 0, 0);
        rig.pose(s.head, Math.sin(t * 0.5) * 0.04 * energy, Math.sin(t * 0.33) * 0.07 * energy, 0);
        const sway = Math.sin(t * 1.2) * 0.02;
        rig.pose(s.upperArmL, 0, 0, -armDown + sway);
        rig.pose(s.upperArmR, 0, 0, armDown - sway);
        // poseLocal: X = elbow bend (curl inward), works regardless of parent rotation
        rig.poseLocal(s.forearmL, 0.35, 0, 0);
        rig.poseLocal(s.forearmR, 0.35, 0, 0);
        const bt = t % 4.0, b = bt < 0.38 ? Math.sin(bt / 0.38 * Math.PI) : 0;
        rig.blink(b);
        rig.morph('Mouth Smile', 0.2);
    },

    // armDown lowers the arms from the horizontal T-pose (rad). pose() uses XYZ
    // world-euler order, so z (down) applies before x — x then swings the hanging
    // arm forward/back instead of twisting a horizontal one.
    //
    // Phase-4 gait v2, Phase-5 facing-corrected. Gait phase w runs BACKWARD in
    // time for +z-facing models (w = −F·t·5·speed): thigh = thighAmp·sin w then
    // sweeps the stance foot front→back while the body advances, and the swing
    // foot travels back→front through the air. (Phase-4 ran w forward, which
    // for +z-facing rigs swept the stance foot forward under the body and
    // swung the airborne foot backward — a universal moonwalk.)
    //   stride: 'auto' scales with measured leg length (stub legs get a shorter,
    //           safer stride); a number keeps the calibrated-override meaning.
    //   bob:    multiplier on pelvis translation — vertical drop at contact
    //           (geometric complement of split straight legs) + lateral weight
    //           shift toward the stance foot.
    //   stab:   anti-slide fore-aft hips shift — partially compensates the
    //           stance foot's backward sweep so the planted foot slides less.
    //   wbal:   support blend 0..0.7 — 0 rides the lower-reaching leg (back
    //           foot floats the full reach gap = natural heel-off, but counted
    //           as float), 0.5 splits the gap (front digs as much as the back
    //           floats), >0.5 rides the back leg.
    // Knees flex through the measured flexion axis (not local X, which is a
    // roll axis on these rigs) and only during swing (bent at passing,
    // released at both contact extremes).
    walk(rig, t, { speed = 1, stride = 'auto', armAmp = 1, armRollOut = 0.22, armDown = rig.armDownAuto ?? 1.4, bob = 1, stab = 0.45, wbal = 0.35 } = {}) {
        const s = rig.semantic;
        const legLen = rig.legLengthAuto ?? 0.5;
        const ax = rig.legAxes ?? { fw: -1 };
        const F = rig.facing ?? 1;
        const strideS = stride === 'auto'
            ? Math.min(1.3, Math.max(0.35, legLen / 0.45))
            : stride;
        const w = -F * t * 5 * speed, A = 0.5 * strideS;
        const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

        const thighAmp = A * 1.2;
        const thighLX = Math.sin(w) * thighAmp, thighRX = -thighLX;
        rig.pose(s.thighL, thighLX, 0, 0);
        rig.pose(s.thighR, thighRX, 0, 0);
        // Knees: bent through passing (cos w > 0 = L swing), straight at both
        // contact extremes so the stance foot stays down through its sweep.
        // The (1 − sin²)² release factor straightens the knee EARLY in late
        // swing — the pelvis support curve assumes the STANCE knee (0.12), so
        // a still-bent knee at touchdown reads as a shorter leg than the curve
        // and the foot gets pressed several mm through the floor. Late swing =
        // approaching the FRONT contact, where sin w ≈ −F (foot at the +F
        // extreme), so the release keys to −F·sin w per side.
        // KNEE_MIN keeps the knee off full extension (no hyperextension flag).
        const KNEE_MIN = 0.12;
        const kneeAmp = Math.min(A * 2.2, 0.9);
        const sw2L = Math.max(0, -F * Math.sin(w)) ** 2, sw2R = Math.max(0, F * Math.sin(w)) ** 2;
        const kneeL = KNEE_MIN + Math.max(0,  Math.cos(w)) * (1 - sw2L) ** 2 * kneeAmp;
        const kneeR = KNEE_MIN + Math.max(0, -Math.cos(w)) * (1 - sw2R) ** 2 * kneeAmp;
        rig._poseMapped(s.shinL, ax.kneeL, kneeL);
        rig._poseMapped(s.shinR, ax.kneeR, kneeR);
        // Pelvis vertical: ride the EMPIRICAL support curve — the pelvis drops
        // by the measured ankle-height change of whichever leg reaches lowest
        // (the no-dig constraint), so the planted foot tracks the ground
        // through the whole sweep. +0.002 hovers soles ~2 mm above digging.
        // (The Phase-4 sin²·(legLen(1−cos)−margin) model assumed vertical rest
        // legs; these chibi rigs are splayed — Bluey 239 mm chain / 161 mm
        // vertical — so it under-dropped mid-stance and over-dropped the back
        // extreme, and could not express the front/back reach asymmetry.)
        // Computed BEFORE the feet: the heel-lift below needs bobY.
        let bobY = 0, heelL = 0, heelR = 0;
        const reachL = rig._legReachAt('L', thighLX), reachR = rig._legReachAt('R', thighRX);
        if (reachL != null && reachR != null) {
            const cL = rig._legReach.L, cR = rig._legReach.R;
            const dL = reachL - cL.y0, dR = reachR - cR.y0;
            // Support height: front/back reaches differ (rest legs aren't
            // fore-aft centered — Bluey +0.27 rad back = 12 mm, front = 5 mm),
            // so full front-support floats the back foot >5 mm through late
            // mid-stance. Blend toward the mean: both feet stay within ~3 mm
            // of ground (inside contact tolerance), matching real weight
            // transfer through the stance.
            const lo = Math.min(dL, dR), hi = Math.max(dL, dR);
            const support = lo + wbal * (hi - lo);
            // hover: base 2 mm + extra at the contact switch, where the pelvis
            // roll and the knee-release residual both bite into clearance
            bobY = -bob * support + (0.002 + 0.003 * Math.abs(Math.sin(w))) * bob;
            // Heel-lift: the unweighting (back) foot would float by soleExcess
            // because its reach shortens faster than the support foot's.
            // Plantarflex its ankle (toes down, heel rises) so the toe stays
            // planted — anatomical late-stance heel-off — instead of hovering.
            // Gated to the stance half so the swing foot keeps clearance.
            const exL = dL * bob + bobY, exR = dR * bob + bobY;   // ≈ sole float
            const cw0 = Math.cos(w);
            if (cw0 < 0 && exL > 0) heelL = clamp(exL / Math.max(cL.footLen, 0.05), 0, 0.65);
            if (cw0 > 0 && exR > 0) heelR = clamp(exR / Math.max(cR.footLen, 0.05), 0, 0.65);
        } else {
            bobY = -bob * Math.max(0, legLen * (1 - Math.cos(thighAmp * Math.sin(w))) - 0.005);
        }
        // Feet: keep the sole level — cancel the accumulated world pitch.
        // Thigh (+x world) and knee (facing-corrected map: +cmd also pitches
        // +x world) now accumulate in the SAME sense, so cancel their sum;
        // heel-lift adds toe-down (world +x) on the unweighting foot.
        rig._poseMapped(s.footL, ax.footL, clamp(-(thighLX + kneeL) + heelL, -0.9, 0.9));
        rig._poseMapped(s.footR, ax.footR, clamp(-(thighRX + kneeR) + heelR, -0.9, 0.9));
        // Arms — opposition phase to the legs (right arm forward with left leg).
        // Phase-5: swing about cross(posedArmDir, facing) so the hand moves
        // purely fore-aft even with rolled-out chibi arms — raw world X is
        // mostly twist there (kuromi V2 walked at 87 mm hand travel vs Bluey
        // 518 mm at the same armAmp). Identical to the old euler for vertical
        // arms, so Bluey-family gait is unchanged.
        // Constant 1.2 → 1.6: tuned strides (0.4–0.55) shrink A, and arm swing
        // is coupled to A — every tuned model walked at ±14° shoulder swing,
        // which reads as frozen on screen (owner: "kuromi V2 arms rigid").
        // Floor 0.45 rad decouples arm swing from leg stride: stub-legged
        // chibi rigs (stride 0.4–0.55 → A·1.6 ≈ 0.32–0.44) still get a
        // readable ±26° swing; long-legged rigs at auto stride exceed the
        // floor and are untouched. The armJ gate still caps armAmp where
        // hands would graze the torso.
        const swA = Math.sin(w) * Math.max(A * 1.6, 0.45) * armAmp;
        rig._poseArmSwing(s.upperArmL, 'L', -(armDown - armRollOut),  swA);
        rig._poseArmSwing(s.upperArmR, 'R',  (armDown - armRollOut), -swA);
        const elbowBase = 0.4;
        const elbowSwingL = Math.max(0, -Math.sin(w + 0.2)) * 0.85 * armAmp;
        const elbowSwingR = Math.max(0,  Math.sin(w + 0.2)) * 0.85 * armAmp;
        rig.poseLocal(s.forearmL, elbowBase + elbowSwingL, 0, 0);
        rig.poseLocal(s.forearmR, elbowBase + elbowSwingR, 0, 0);
        // Lateral: settle over the stance foot (L stance while cos w < 0).
        const shiftX = -(rig._stanceSign ?? 1) * 0.04 * legLen * bob * Math.cos(w);
        // Fore-aft anti-slide: track the stance foot's sweep so the planted
        // foot moves less over ground. Blended to zero at the contact switch
        // (g) where the reference foot changes and foot speed is ~0 anyway.
        const cw = Math.cos(w), acw = Math.abs(cw);
        const g = clamp((acw - 0.1) / 0.35, 0, 1);
        const shiftZ = -stab * legLen * Math.sin(thighAmp * Math.sin(w)) * Math.sign(cw) * g;
        rig._driveHips(bobY, shiftX, shiftZ);
        // Residual rotational sway, gentler now that translation carries the
        // motion. Roll 0.03 rad: at the contact switch (|sin w|=1) each pelvis
        // socket drops ~hipHalfWidth·roll ≈ 1–2 mm on the incoming-stance
        // side — 0.05 read as foot digging once bob rides the measured
        // support curve instead of Phase-4's fat constant margin.
        rig.pose(s.hips, 0, Math.sin(w) * 0.10, Math.sin(w) * 0.03);
        if (s.spine.at(-1)) rig.pose(s.spine.at(-1), 0.05, -Math.sin(w) * 0.1, 0);
        rig.pose(s.head, Math.sin(t * 0.5) * 0.04, Math.sin(t * 0.33) * 0.06, 0);
        s.tail.forEach((n, i) => rig.pose(n, 0, Math.sin(t * 6 * speed - i * 0.7) * (0.08 + i * 0.07), 0));
        const bt = t % 4.0, b = bt < 0.38 ? Math.sin(bt / 0.38 * Math.PI) : 0;
        rig.blink(b);
    },

    run(rig, t, opts = {}) {
        const stride = (opts.stride === undefined || opts.stride === 'auto') ? 'auto' : opts.stride * 1.5;
        CLIPS.walk(rig, t, { ...opts, speed: (opts.speed ?? 1) * 1.9, stride });
        const s = rig.semantic;
        if (s.spine[0]) rig.pose(s.spine[0], 0.22, 0, 0);   // lean forward
        rig.pose(s.head, -0.12, 0, 0);
    },

    wave(rig, t, { side = 'R', raiseAmt = 0.65, elbowBend = 1.6, armDown = rig.armDownAuto ?? 1.4 } = {}) {
        const s = rig.semantic;
        const up = side === 'R' ? s.upperArmR : s.upperArmL;
        const fo = side === 'R' ? s.forearmR : s.forearmL;
        const ha = side === 'R' ? s.handR : s.handL;
        const sign = side === 'R' ? -1 : 1;
        const wag = Math.sin(t * 3.5);
        CLIPS.idle(rig, t, { armDown });
        rig.pose(s.hips, 0, 0, 0);
        if (s.spine[0]) rig.pose(s.spine[0], Math.sin(t * 1.5) * 0.02, 0, 0);
        rig.pose(up, -0.3, sign * 0.3, sign * raiseAmt);
        rig.poseLocal(fo, elbowBend, 0, wag * 0.15);
        rig.poseLocal(ha, 0, -sign * 0.8, wag * 0.35);
        rig.morph('Mouth Smile', 0.7);
        rig.viseme('A', 0.2);
    },

    walkWave(rig, t, opts = {}) {
        const { side = 'R', raiseAmt = 0.65, elbowBend = 1.6 } = opts;
        CLIPS.walk(rig, t, opts);
        const s = rig.semantic;
        const up   = side === 'R' ? s.upperArmR : s.upperArmL;
        const fo   = side === 'R' ? s.forearmR  : s.forearmL;
        const ha   = side === 'R' ? s.handR     : s.handL;
        const sign = side === 'R' ? -1 : 1;
        const wag  = Math.sin(t * 3.5);
        rig.pose(up, -0.3, sign * 0.3, sign * raiseAmt);
        rig.poseLocal(fo, elbowBend, 0, wag * 0.15);
        rig.poseLocal(ha, 0, -sign * 0.8, wag * 0.35);
        rig.morph('Mouth Smile', 0.6);
    },

    talk(rig, t, { energy = 1, armDown = rig.armDownAuto ?? 1.4 } = {}) {
        // armDown is forwarded to idle() so calibration can reach the arms
        // during talk (previously only {energy} was forwarded — talk could not
        // be tuned). Calls without params behave exactly as before.
        CLIPS.idle(rig, t, { energy: 1.6, armDown });
        const vowels = ['A', 'E', 'I', 'O', 'U', null];
        const slot = Math.floor(t / 0.55) % vowels.length;
        const amt = Math.sin(((t % 0.55) / 0.55) * Math.PI) * 0.9 * energy;
        for (const v of vowels) if (v) rig.viseme(v, v === vowels[slot] ? amt : 0);
        rig.pose(rig.semantic.head, Math.sin(t * 2.1) * 0.05, Math.sin(t * 1.3) * 0.08, Math.sin(t * 1.7) * 0.03);
        rig.morph('BrowHappy', Math.max(0, Math.sin(t * 0.9)) * 0.6);
    },

    dance(rig, t, { bpm = 120, armAmp = 1, armRollOut = 0 } = {}) {
        const s = rig.semantic, beat = t * (bpm / 60) * Math.PI;
        const bounce = Math.abs(Math.sin(beat));
        rig._driveHips(0, 0); // pelvis translation belongs to walk/jump only
        rig.pose(s.hips, 0, Math.sin(beat) * 0.25, Math.sin(beat) * 0.15);
        if (s.spine[0]) rig.pose(s.spine[0], 0.05, -Math.sin(beat) * 0.12, 0);
        rig.pose(s.head, Math.sin(beat * 2) * 0.08, 0, Math.sin(beat) * 0.12);
        rig.pose(s.upperArmL, 0, 0, (0.9 + Math.sin(beat) * 0.6) * armAmp + armRollOut);
        rig.pose(s.upperArmR, 0, 0, (-0.9 + Math.sin(beat) * 0.6) * armAmp - armRollOut);
        rig.poseLocal(s.forearmL, (0.4 + bounce * 0.5) * armAmp, 0, 0);
        rig.poseLocal(s.forearmR, (0.4 + bounce * 0.5) * armAmp, 0, 0);
        rig.pose(s.thighL, -bounce * 0.15, 0, 0.08);
        rig.pose(s.thighR, -bounce * 0.15, 0, -0.08);
        rig._poseMapped(s.shinL, rig.legAxes?.kneeL, bounce * 0.3);
        rig._poseMapped(s.shinR, rig.legAxes?.kneeR, bounce * 0.3);
        s.tail.forEach((n, i) => rig.pose(n, 0, Math.sin(beat * 2 - i * 0.7) * (0.1 + i * 0.09), 0));
        rig.morph('Mouth Smile', 0.8);
        const bt = t % 3.2, b = bt < 0.35 ? Math.sin(bt / 0.35 * Math.PI) : 0;
        rig.blink(b);
    },

    // Phase-4: crouch/air amplitudes scale with measured leg length so stub-leg
    // models squat instead of folding into themselves, and the root lift returns
    // to baseRootPos (captured after normalize) instead of stomping the
    // grounding offset to y=0. Crouch also sinks the pelvis a touch via
    // _driveHips — a real squat instead of thigh-rotation-only.
    jump(rig, t, { height = 1, armAmp = 1 } = {}) {
        const s = rig.semantic, cycle = t % 1.6, ph = cycle / 1.6;
        const legLen = rig.legLengthAuto ?? 0.5;
        const legScale = Math.min(1.2, Math.max(0.5, legLen / 0.45));
        let crouch = 0, air = 0;
        if (ph < 0.25) crouch = Math.sin(ph / 0.25 * Math.PI / 2);            // wind up
        else if (ph < 0.55) { air = Math.sin((ph - 0.25) / 0.3 * Math.PI); }  // airborne
        else if (ph < 0.75) crouch = Math.sin((0.75 - ph) / 0.2 * Math.PI / 2) * 0.6; // land
        rig.pose(s.thighL, (-crouch * 0.8 + air * 0.4) * legScale, 0, 0);
        rig.pose(s.thighR, (-crouch * 0.8 + air * 0.4) * legScale, 0, 0);
        // Knees flex through the measured axis; tuck slightly in flight instead
        // of the old −air·0.2 hyperextension; never fully straight (defect E).
        const kneeAmt = Math.max(0.05, crouch * 1.1 * legScale + air * 0.3);
        rig._poseMapped(s.shinL, rig.legAxes?.kneeL, kneeAmt);
        rig._poseMapped(s.shinR, rig.legAxes?.kneeR, kneeAmt);
        // Ankles: keep the sole level through the crouch — cancel the
        // accumulated world pitch (thigh + knee), exactly like the walk
        // flatten. (Phase-4's +crouch·0.75 term was tuned for the inverted
        // knee map; with anatomical knees it pressed the toes ~10 mm through
        // the floor on landing.) Plus toe-point in flight (+cmd = toes down).
        const fdir = rig.facing ?? 1;
        const thighP = (-crouch * 0.8 + air * 0.4) * legScale;
        const ankle = Math.min(0.6, Math.max(-0.6, -(thighP + kneeAmt))) + fdir * air * 0.5;
        rig._poseMapped(s.footL, rig.legAxes?.footL, ankle);
        rig._poseMapped(s.footR, rig.legAxes?.footR, ankle);
        if (s.spine[0]) rig.pose(s.spine[0], (crouch * 0.35 - air * 0.15) * legScale, 0, 0);
        rig.pose(s.upperArmL, 0, 0, (crouch * -0.6 + air * 1.7) * armAmp);
        rig.pose(s.upperArmR, 0, 0, (crouch * 0.6 - air * 1.7) * armAmp);
        rig.pose(s.head, -crouch * 0.2 + air * 0.15, 0, 0);
        rig._driveHips(-crouch * 0.03 * legLen, 0);
        const root = rig.root;
        const baseY = rig.baseRootPos ? rig.baseRootPos.y : 0;
        root.position.y = baseY + Math.max(0, air) * 0.55 * height * root.scale.y * legScale;
    }
};

// ---------------------------------------------------------------------------
// CollisionGuard — geometric anti-clip for arms.
//
// At load it samples the skinned mesh for vertices weighted to the torso bones
// (Hips + Spine chain) and fits an ellipsoid around them in hips-local space.
// Every frame it measures how far the forearm/hand joints have sunk inside that
// ellipsoid and feeds a corrective outward roll onto the upper arm — a small
// feedback spring, so corrections fade in/out smoothly and vanish when clear.
// Runs automatically inside rig.update(); disable with rig.collision.enabled=false
// or construct the rig with { collision: false }.
// ---------------------------------------------------------------------------
export class CollisionGuard {
    constructor(rig) {
        this.rig = rig;
        this.enabled = true;
        this.gain = 10;      // how fast correction grows while penetrating (rad/s per unit depth)
        this.decay = 2.5;    // how fast it relaxes once clear (rad/s)
        this.max = 1.1;      // correction cap (rad)
        this.holdFor = 1.2;  // s to hold a correction after the last hit — looping clips
                             // re-offend every cycle, so don't give ground between beats
        this._cor = { L: 0, R: 0 };
        this._hit = { L: -1, R: -1 };
        this._suppressUntil = 0;
        const THREE = rig.THREE;
        this._v = new THREE.Vector3();
        this._v2 = new THREE.Vector3();
        this._inv = new THREE.Matrix4();
        this._e = new THREE.Euler();
        this._q = new THREE.Quaternion();
        this._q2 = new THREE.Quaternion();
        this._build();
    }

    // Call from a clip to pause guard for `seconds` — prevents jelly-shake
    // when the arm is intentionally near the head (wave, dance raise, etc.)
    suppress(seconds) { this._suppressUntil = this.rig._t + seconds; }

    _build() {
        const rig = this.rig, s = rig.semantic;
        this.volumes = [];
        rig.root.updateMatrixWorld(true);
        // torso: verts weighted to Hips + Spine chain, anchored to the hips bone
        this._addVolume([s.hips, ...s.spine].filter(Boolean), 0.92);
        // head: these cartoon heads are huge relative to the head bone, so raised
        // arms (dance, wave) clip into them — verts weighted to Head/Snout/lids.
        // Ear-chain verts are EXCLUDED: on long-eared models (Cinnamoroll) they
        // inflate the ellipsoid sideways until it engulfs the T-pose arms, making
        // every head measurement baseline-dominated. Accessory chains (Bow/Hat/
        // Dress) were never in this set either.
        this._addVolume([s.head, s.snout,
            ...s.lidTopL, ...s.lidTopR, ...s.lidBotL, ...s.lidBotR].filter(Boolean), 0.90);
        if (!this.volumes.length) this.enabled = false;
    }

    // Fit an ellipsoid around all mesh verts whose dominant skin weight belongs to
    // `names`, stored in the first bone's local space so it follows that bone.
    _addVolume(names, shrink) {
        const rig = this.rig, THREE = rig.THREE;
        const anchor = names[0] && rig.bones[names[0]]?.bone;
        if (!anchor) return;
        const set = new Set(names);
        const inv = new THREE.Matrix4().copy(anchor.matrixWorld).invert();
        const min = new THREE.Vector3(Infinity, Infinity, Infinity);
        const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
        const v = new THREE.Vector3();
        let n = 0;
        rig.root.traverse((mesh) => {
            if (!mesh.isSkinnedMesh) return;
            const pos = mesh.geometry.attributes.position;
            const si = mesh.geometry.attributes.skinIndex;
            const sw = mesh.geometry.attributes.skinWeight;
            if (!pos || !si || !sw) return;
            const step = Math.max(1, Math.floor(pos.count / 5000));
            for (let i = 0; i < pos.count; i += step) {
                let bi = si.getX(i), bw = sw.getX(i);
                if (sw.getY(i) > bw) { bw = sw.getY(i); bi = si.getY(i); }
                if (sw.getZ(i) > bw) { bw = sw.getZ(i); bi = si.getZ(i); }
                if (sw.getW(i) > bw) { bw = sw.getW(i); bi = si.getW(i); }
                if (!set.has(mesh.skeleton.bones[bi]?.name)) continue;
                // getVertexPosition applies bone skinning — same space the probes live in
                if (mesh.getVertexPosition) mesh.getVertexPosition(i, v);
                else v.fromBufferAttribute(pos, i);
                v.applyMatrix4(mesh.matrixWorld).applyMatrix4(inv);
                min.min(v); max.max(v);
                n++;
            }
        });
        if (n < 20) return;
        const radii = max.clone().sub(min).multiplyScalar(0.5 * shrink);
        radii.x = Math.max(radii.x, 1e-4);
        radii.y = Math.max(radii.y, 1e-4);
        radii.z = Math.max(radii.z, 1e-4);
        this.volumes.push({
            node: anchor,
            center: min.clone().add(max).multiplyScalar(0.5),
            radii,
            inv: new THREE.Matrix4()
        });
    }

    // deepest penetration (0..1) of a world point into any volume; 0 when outside all
    _depth(p) {
        let d = 0;
        for (const vol of this.volumes) {
            this._v2.copy(p).applyMatrix4(vol.inv).sub(vol.center).divide(vol.radii);
            d = Math.max(d, 1 - this._v2.length());
        }
        return Math.max(0, d);
    }

    update(dt) {
        if (!this.volumes?.length) return;
        if (this.rig._t < this._suppressUntil) return;
        const rig = this.rig, s = rig.semantic, THREE = rig.THREE;
        for (const vol of this.volumes) {
            vol.node.updateWorldMatrix(true, false);
            vol.inv.copy(vol.node.matrixWorld).invert();
        }
        for (const side of ['L', 'R']) {
            const up = rig.bones[s['upperArm' + side]];
            const fo = rig.bones[s['forearm' + side]]?.bone;
            const ha = rig.bones[s['hand' + side]]?.bone;
            if (!up) continue;
            // deepest-penetrating probe point and which volume it hit
            let pen = 0, probe = null, volHit = null;
            const consider = (p) => {
                for (const vol of this.volumes) {
                    this._v2.copy(p).applyMatrix4(vol.inv).sub(vol.center).divide(vol.radii);
                    const d = 1 - this._v2.length();
                    if (d > pen) { pen = d; probe = p.clone(); volHit = vol; }
                }
            };
            if (fo) consider(fo.getWorldPosition(this._v));
            if (ha) {
                const hp = ha.getWorldPosition(this._v).clone();
                consider(hp);
                if (fo) { // extrapolated hand tip
                    fo.getWorldPosition(this._v);
                    consider(hp.clone().add(hp.clone().sub(this._v).multiplyScalar(0.9)));
                }
            }
            const c = this._cor;
            if (pen > 0 && probe) {
                // Escape direction, not a fixed "outward": the correction is a world-Z
                // roll at the shoulder, so the probe moves along ẑ × (probe - shoulder).
                // Roll positive if that motion points away from the volume's center
                // (lifts a low arm off the torso, but LOWERS a raised arm off the head).
                const sh = up.bone.getWorldPosition(this._v).clone();
                const vel = new THREE.Vector3(0, 0, 1).cross(probe.clone().sub(sh));
                const away = probe.sub(volHit.node.localToWorld(volHit.center.clone()));
                const dir = vel.dot(away) >= 0 ? 1 : -1;
                c[side] = Math.max(-this.max, Math.min(this.max, c[side] + dir * pen * this.gain * dt));
                this._hit[side] = rig._t;
            } else if (rig._t - this._hit[side] > this.holdFor) {
                const d = this.decay * dt;
                c[side] = Math.abs(c[side]) <= d ? 0 : c[side] - Math.sign(c[side]) * d;
            }
            if (Math.abs(c[side]) > 1e-3) {
                this._e.set(0, 0, c[side], 'XYZ');
                this._q.setFromEuler(this._e);
                this._q2.copy(up.restWorldInv).multiply(this._q).multiply(up.restWorld);
                up.target.multiply(this._q2);   // compose on top of whatever the clip set
                up.lastSet = rig._t;
            }
        }
    }
}
