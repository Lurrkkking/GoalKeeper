#!/usr/bin/env python3
"""Dump all active collision geom friction/contact params from a MuJoCo XML."""
import sys, mujoco, numpy as np

xml = sys.argv[1] if len(sys.argv) > 1 else "q1_abi_D2.xml"

m = mujoco.MjModel.from_xml_path(xml)

GEOM_TYPES = {0: 'plane', 2: 'sphere', 3: 'capsule', 5: 'cylinder', 6: 'box', 7: 'mesh'}
BODY_MAP = {m.body(i).name: i for i in range(m.nbody)}

print(f"XML: {xml}\n")

# Check visual meshes
mesh_issues = 0
for i in range(m.ngeom):
    if m.geom(i).type == 7:  # mesh
        if m.geom_contype[i] != 0 or m.geom_conaffinity[i] != 0:
            print(f"WARNING: mesh geom[{i}] {m.body(m.geom_bodyid[i]).name} has contype={m.geom_contype[i]} conaff={m.geom_conaffinity[i]}")
            mesh_issues += 1
if mesh_issues == 0:
    print("OK: all visual meshes have contype=0 conaffinity=0\n")

# Active collision geoms
print(f"{'id':>3s} {'name':>15s} {'body':>25s} {'type':>8s} {'ct':>2s} {'ca':>2s} {'cd':>2s} {'pr':>2s} {'friction[slide,spin,roll]':>30s} {'solref':>12s} {'solimp':>16s}")
print("-"*150)

for i in range(m.ngeom):
    ct = m.geom_contype[i]
    ca = m.geom_conaffinity[i]
    if ct == 0 and ca == 0:
        continue
    gtype = GEOM_TYPES.get(int(m.geom(i).type), str(m.geom(i).type))
    body = m.body(m.geom_bodyid[i]).name
    name = m.geom(i).name or "(unnamed)"
    fr = m.geom_friction[i]
    sr = m.geom_solref[i]
    si = m.geom_solimp[i]
    cd = m.geom_condim[i]
    pr = m.geom_priority[i]
    print(f"{i:3d} {name:>15s} {body:>25s} {gtype:>8s} {ct:2d} {ca:2d} {cd:2d} {pr:2d} "
          f"[{fr[0]:.3f}, {fr[1]:.3f}, {fr[2]:.4f}] "
          f"[{sr[0]:.3f}, {sr[1]:.1f}] "
          f"[{si[0]:.3f}, {si[1]:.3f}, {si[2]:.3f}]")

# Summary by category
print("\n=== Summary by category ===")
cats = {'floor': [], 'sole': [], 'body': [], 'ball': []}
for i in range(m.ngeom):
    ct = m.geom_contype[i]
    if ct == 0: continue
    body = m.body(m.geom_bodyid[i]).name
    if ct == 4: cats['floor'].append(i)
    elif ct == 1: cats['sole'].append(i)
    elif ct == 8: cats['body'].append(i)
    elif ct == 2: cats['ball'].append(i)

for cat, idxs in cats.items():
    if not idxs: continue
    i = idxs[0]
    fr = m.geom_friction[i]
    sr = m.geom_solref[i]
    si = m.geom_solimp[i]
    print(f"  {cat:6s} (n={len(idxs)}): friction=[{fr[0]:.3f},{fr[1]:.3f},{fr[2]:.4f}] "
          f"condim={m.geom_condim[i]} priority={m.geom_priority[i]} "
          f"solref=[{sr[0]:.3f},{sr[1]:.1f}] solimp=[{si[0]:.3f},{si[1]:.3f},{si[2]:.3f}]")
