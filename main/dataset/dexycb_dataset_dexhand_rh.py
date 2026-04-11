import os
import pickle
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes

from main.dataset.transform import rotmat_to_aa
from .base import ManipData
from .decorators import register_manipdata


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@register_manipdata("dexycb_rh")
class DexYCBDatasetDexHandRH(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/DexYCB/processed",
        split: str = "all",
        skip: int = 2,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        point_track_k: int = 32,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        self.point_track_k = int(point_track_k)

        if os.path.isdir(self.data_dir):
            self.data_pathes = sorted(
                [
                    os.path.join(self.data_dir, p)
                    for p in os.listdir(self.data_dir)
                    if p.endswith(".pkl") or p.endswith(".pickle")
                ]
            )
        else:
            self.data_pathes = []

    def _parse_index(self, index) -> Tuple[str, int]:
        if isinstance(index, str):
            if not index.startswith("d"):
                raise ValueError(f"DexYCB index must start with 'd', got {index}.")
            body = index[1:]
            if "@" in body:
                seq_id, stage_str = body.split("@", 1)
                return seq_id, int(stage_str)
            return body, 0
        raise ValueError(f"Unsupported DexYCB index type: {type(index)}")

    def _resolve_sample_path(self, seq_id: str, stage: int) -> str:
        candidates = [
            os.path.join(self.data_dir, f"{seq_id}@{stage}.pkl"),
            os.path.join(self.data_dir, f"{seq_id}_{stage}.pkl"),
            os.path.join(self.data_dir, f"{seq_id}.pkl"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

        if seq_id.isdigit() and self.data_pathes:
            i = int(seq_id)
            if 0 <= i < len(self.data_pathes):
                return self.data_pathes[i]

        raise FileNotFoundError(
            f"Cannot resolve DexYCB sample for seq={seq_id}, stage={stage} under {self.data_dir}."
        )

    def _resolve_obj_verts(self, payload: Dict) -> torch.Tensor:
        if "obj_verts" in payload:
            return _to_tensor(payload["obj_verts"], self.device)

        mesh_path = payload.get("obj_mesh_path", None)
        if mesh_path is None:
            raise KeyError("DexYCB sample requires either 'obj_verts' or 'obj_mesh_path'.")
        mesh = trimesh.load(mesh_path, process=False)
        mesh_struct = Meshes(
            verts=torch.from_numpy(mesh.vertices[None].astype(np.float32)),
            faces=torch.from_numpy(mesh.faces[None].astype(np.int64)),
        )
        return self.random_sampling_pc(mesh_struct)

    def _resolve_mano_joints(self, payload: Dict) -> Dict[str, torch.Tensor]:
        if "mano_joints" in payload and isinstance(payload["mano_joints"], dict):
            return {k: _to_tensor(v, self.device) for k, v in payload["mano_joints"].items()}

        joints = payload.get("mano_joints", payload.get("hand_joints", None))
        if joints is None:
            raise KeyError("DexYCB sample requires 'mano_joints' (dict) or 'hand_joints' ([T,J,3]).")
        joints = _to_tensor(joints, self.device)
        if joints.ndim != 3 or joints.shape[-1] != 3:
            raise ValueError(f"mano_joints tensor must be [T,J,3], got {list(joints.shape)}")

        ordered_names = [
            self.dexhand.to_hand(j_name)[0]
            for j_name in self.dexhand.body_names
            if self.dexhand.to_hand(j_name)[0] != "wrist"
        ]
        n_need = len(ordered_names)
        if joints.shape[1] < n_need:
            pad = joints[:, -1:, :].repeat(1, n_need - joints.shape[1], 1)
            joints = torch.cat([joints, pad], dim=1)
        elif joints.shape[1] > n_need:
            joints = joints[:, :n_need, :]

        return {name: joints[:, i] for i, name in enumerate(ordered_names)}

    def _resolve_wrist_rot_aa(self, payload: Dict) -> torch.Tensor:
        wrist_rot = payload.get("wrist_rot", payload.get("hand_rot", None))
        if wrist_rot is None:
            raise KeyError("DexYCB sample requires 'wrist_rot' or 'hand_rot'.")
        wrist_rot = _to_tensor(wrist_rot, self.device)
        if wrist_rot.ndim == 2 and wrist_rot.shape[-1] == 3:
            return wrist_rot
        if wrist_rot.ndim == 3 and wrist_rot.shape[-2:] == (3, 3):
            return rotmat_to_aa(wrist_rot)
        raise ValueError(f"Unsupported wrist rotation shape: {list(wrist_rot.shape)}")

    def _resolve_obj_trajectory(self, payload: Dict) -> torch.Tensor:
        obj_trajectory = payload.get("obj_trajectory", payload.get("object_trajectory", None))
        if obj_trajectory is None:
            raise KeyError("DexYCB sample requires 'obj_trajectory' or 'object_trajectory'.")
        obj_trajectory = _to_tensor(obj_trajectory, self.device)
        if obj_trajectory.ndim != 3 or obj_trajectory.shape[-2:] != (4, 4):
            raise ValueError(f"obj_trajectory must be [T,4,4], got {list(obj_trajectory.shape)}")
        return obj_trajectory

    def _build_point_tracks(self, data: Dict, k: int):
        base_points = data["obj_verts"]
        if base_points.shape[0] < k:
            pad = base_points[-1:].repeat(k - base_points.shape[0], 1)
            base_points = torch.cat([base_points, pad], dim=0)
        else:
            base_points = base_points[:k]

        obj_traj = data["obj_trajectory"]  # [T,4,4] in gym/world frame after process_data
        points_world = (obj_traj[:, :3, :3] @ base_points.T.unsqueeze(0)).transpose(-1, -2) + obj_traj[:, :3, 3][:, None]
        points_target = torch.cat([points_world[1:], points_world[-1:]], dim=0)
        points_mask = torch.ones(points_world.shape[0], points_world.shape[1], device=self.device, dtype=torch.float32)

        data["obj_points_t"] = points_world
        data["obj_points_target_t"] = points_target
        data["obj_points_mask_t"] = points_mask

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        seq_id, stage = self._parse_index(index)
        sample_path = self._resolve_sample_path(seq_id, stage)
        payload = _load_pickle(sample_path)

        obj_trajectory = self._resolve_obj_trajectory(payload)
        wrist_pos_raw = payload.get("wrist_pos", payload.get("hand_pos", None))
        if wrist_pos_raw is None:
            raise KeyError("DexYCB sample requires 'wrist_pos' or 'hand_pos'.")
        wrist_pos = _to_tensor(wrist_pos_raw, self.device)
        wrist_rot = self._resolve_wrist_rot_aa(payload)
        mano_joints = self._resolve_mano_joints(payload)
        obj_verts = self._resolve_obj_verts(payload)

        obj_mesh_path = payload.get("obj_mesh_path", "")
        obj_urdf_path = payload.get("obj_urdf_path", "")
        if not obj_urdf_path and obj_mesh_path:
            obj_urdf_path = os.path.splitext(obj_mesh_path)[0] + ".urdf"

        data = {
            "data_path": sample_path,
            "obj_id": str(payload.get("obj_id", seq_id)),
            "obj_mesh_path": obj_mesh_path,
            "obj_verts": obj_verts,
            "obj_urdf_path": obj_urdf_path,
            "obj_trajectory": obj_trajectory[:: self.skip],
            "scene_objs": payload.get("scene_objs", []),
            "wrist_pos": wrist_pos[:: self.skip],
            "wrist_rot": wrist_rot[:: self.skip],
            "mano_joints": {k: v[:: self.skip] for k, v in mano_joints.items()},
        }

        self.process_data(data, 0, data["obj_verts"])
        self._build_point_tracks(data, self.point_track_k)

        opt_root = f"data/retargeting/DexYCB/mano2{str(self.dexhand)}"
        opt_name = os.path.splitext(os.path.basename(sample_path))[0] + ".pkl"
        self.load_retargeted_data(data, os.path.join(opt_root, opt_name))
        return data
