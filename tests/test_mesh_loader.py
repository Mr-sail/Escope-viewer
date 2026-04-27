from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.mesh_loader import load_mesh


ASCII_STL = """\
solid triangle
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid triangle
"""


class MeshLoaderTests(unittest.TestCase):
    def test_load_ascii_stl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "triangle.stl"
            path.write_text(ASCII_STL, encoding="utf-8")

            mesh = load_mesh(path)

        self.assertEqual(mesh.vertices.shape, (3, 3))
        self.assertEqual(mesh.faces.shape, (1, 3))
        np.testing.assert_allclose(mesh.vertices[2], np.array([0.0, 1.0, 0.0]))

    def test_load_binary_stl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "triangle.stl"
            header = b"binary-stl".ljust(80, b"\0")
            triangle_count = struct.pack("<I", 1)
            triangle = struct.pack(
                "<12fH",
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0,
            )
            path.write_bytes(header + triangle_count + triangle)

            mesh = load_mesh(path)

        self.assertEqual(mesh.vertices.shape, (3, 3))
        self.assertEqual(mesh.faces.tolist(), [[0, 1, 2]])
        np.testing.assert_allclose(mesh.vertices[1], np.array([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
