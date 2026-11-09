import numpy as np
for scene, frame in [(0, 0), (37, 0), (37, 100)]:
    g = np.load(f'/home/jihoon/data/occ3d_waymo/validation/{scene:03d}/{frame:03d}_04.npz')
    lab = g['voxel_label']
    print(f'scene {scene} frame {frame}')
    for k in ['infov', 'origin_voxel_state', 'final_voxel_state']:
        a = g[k]
        print(f'  {k}: dtype {a.dtype}, values {dict(zip(*[x.tolist() for x in np.unique(a, return_counts=True)]))}')
    m = g['infov'].astype(bool) & g['final_voxel_state'].astype(bool)
    print('  mask voxels', m.sum(), ' free among them %.1f%%' % (100 * (lab[m] == 23).mean()))
    for v in np.unique(g['final_voxel_state']):
        sel = g['infov'].astype(bool) & (g['final_voxel_state'] == v)
        print(f'  final_state={v}: voxels {sel.sum()}, free {100 * (lab[sel] == 23).mean():.1f}%')
