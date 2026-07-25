// apply_calibration.js — auto-apply calibrations baked into a GLB.
//
// The rig engine never reads GLB extras by itself, so calibrations baked via
// bake_calibrations.mjs (root extras.calibration) would otherwise be ignored.
// Call this right after constructing the RigCharacter:
//
//   const gltf = await new GLTFLoader().loadAsync('Model_fixed.glb');
//   const rig = new RigCharacter(gltf.scene, THREE);
//   applyCalibration(gltf, rig);        // ← one line
//
// Returns the applied calibration (or null), so callers can show status.

export function applyCalibration(gltf, rig) {
    const cal = gltf?.parser?.json?.extras?.calibration
             ?? gltf?.userData?.calibration
             ?? null;
    if (cal && rig) rig.calibrate(cal);
    return cal;
}
