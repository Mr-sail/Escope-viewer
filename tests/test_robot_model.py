from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.robot_model import RobotModelError, load_robot_model


SIMPLE_URDF = """\
<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link" />
  <link name="link1" />
  <link name="tool0" />
  <joint name="joint1" type="revolute">
    <parent link="base_link" />
    <child link="link1" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
  <joint name="joint2" type="fixed">
    <parent link="link1" />
    <child link="tool0" />
    <origin xyz="1 0 0" rpy="0 0 0" />
  </joint>
</robot>
"""


class RobotModelTests(unittest.TestCase):
    def test_load_robot_model_from_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simple.urdf"
            path.write_text(SIMPLE_URDF, encoding="utf-8")

            model = load_robot_model(path)

        self.assertEqual(model.name, "simple_arm")
        self.assertEqual(model.root_link, "base_link")
        self.assertEqual([joint.name for joint in model.movable_joints], ["joint1"])

    def test_compute_segments_applies_joint_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simple.urdf"
            path.write_text(SIMPLE_URDF, encoding="utf-8")
            model = load_robot_model(path)

        segments = model.compute_segments({"joint1": np.pi / 2.0})

        self.assertEqual(len(segments), 2)
        first_segment_end = segments[0][1]
        second_segment_end = segments[1][1]
        np.testing.assert_allclose(first_segment_end, np.array([1.0, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(second_segment_end, np.array([1.0, 1.0, 0.0]), atol=1e-6)

    def test_invalid_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.urdf"
            path.write_text("<robot>", encoding="utf-8")

            with self.assertRaises(RobotModelError):
                load_robot_model(path)


if __name__ == "__main__":
    unittest.main()
