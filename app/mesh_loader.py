from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


class MeshLoadError(ValueError):
    """Raised when a mesh file cannot be loaded for robot rendering."""


@dataclass(frozen=True)
class LoadedMesh:
    vertices: np.ndarray
    faces: np.ndarray


def load_mesh(path: str | Path) -> LoadedMesh:
    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.exists():
        raise MeshLoadError(f"mesh 文件不存在: {mesh_path}")
    if not mesh_path.is_file():
        raise MeshLoadError(f"mesh 路径不是文件: {mesh_path}")

    suffix = mesh_path.suffix.lower()
    if suffix == ".stl":
        return _load_stl(mesh_path)
    return _load_with_trimesh(mesh_path)


def _make_mesh(vertices: np.ndarray) -> LoadedMesh:
    if vertices.size == 0 or vertices.shape[0] % 3 != 0:
        raise MeshLoadError("mesh 没有有效三角面。")
    faces = np.arange(vertices.shape[0], dtype=np.uint32).reshape(-1, 3)
    return LoadedMesh(vertices=vertices.astype(np.float32, copy=False), faces=faces)


def _load_stl(path: Path) -> LoadedMesh:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack("<I", data[80:84])[0]
        expected_size = 84 + triangle_count * 50
        if triangle_count > 0 and expected_size == len(data):
            dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            )
            records = np.frombuffer(data, dtype=dtype, count=triangle_count, offset=84)
            return _make_mesh(records["vertices"].reshape(-1, 3).copy())

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:
        raise MeshLoadError(f"STL 读取失败: {exc}") from exc

    vertices: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 4 or parts[0].lower() != "vertex":
            continue
        try:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError as exc:
            raise MeshLoadError(f"STL 顶点坐标无效: {line.strip()!r}") from exc

    if not vertices:
        raise MeshLoadError(f"无法从 STL 中读取三角面: {path}")
    return _make_mesh(np.array(vertices, dtype=np.float32))


def _load_with_trimesh(path: Path) -> LoadedMesh:
    try:
        import trimesh  # type: ignore
    except Exception as exc:
        raise MeshLoadError(
            f"加载 {path.suffix or 'mesh'} 需要安装 trimesh；DAE 还需要 pycollada。"
        ) from exc

    try:
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except Exception as exc:
        raise MeshLoadError(f"mesh 读取失败: {exc}") from exc

    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            pass

    vertices = np.asarray(getattr(mesh, "vertices", None), dtype=np.float32)
    faces = np.asarray(getattr(mesh, "faces", None), dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size == 0:
        raise MeshLoadError(f"mesh 没有有效顶点: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        raise MeshLoadError(f"mesh 没有有效三角面: {path}")

    return LoadedMesh(vertices=vertices, faces=faces)
