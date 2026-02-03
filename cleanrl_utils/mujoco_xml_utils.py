from __future__ import annotations

import os
import pathlib
import xml.etree.ElementTree as ET
from typing import Optional

import gymnasium as gym


def _float_list(text: str):
    return [float(x) for x in text.strip().split()]


def _format_floats(values):
    return " ".join(f"{v:.8g}" for v in values)


def locate_mujoco_xml(env_id: str) -> str:
    env = gym.make(env_id)
    try:
        if hasattr(env.unwrapped, "model_path"):
            return str(env.unwrapped.model_path)
        if hasattr(env.unwrapped, "model") and hasattr(env.unwrapped.model, "xml_path"):
            return str(env.unwrapped.model.xml_path)
        raise AttributeError(f"Could not locate XML path for env_id={env_id}")
    finally:
        env.close()


def perturb_mujoco_xml(
    base_xml_path: str,
    out_dir: str,
    *,
    run_name: str,
    body_mass_scale: float = 1.0,
    geom_friction_scale: float = 1.0,
    joint_damping_scale: float = 1.0,
    actuator_gain_scale: float = 1.0,
    actuator_bias_scale: float = 1.0,
) -> str:
    base_xml_path = os.path.abspath(base_xml_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    if body_mass_scale != 1.0:
        for body in root.iter("body"):
            if "mass" in body.attrib:
                body.attrib["mass"] = str(float(body.attrib["mass"]) * body_mass_scale)

    if geom_friction_scale != 1.0:
        for geom in root.iter("geom"):
            if "friction" in geom.attrib:
                fr = _float_list(geom.attrib["friction"])
                fr = [v * geom_friction_scale for v in fr]
                geom.attrib["friction"] = _format_floats(fr)

    if joint_damping_scale != 1.0:
        for joint in root.iter("joint"):
            if "damping" in joint.attrib:
                joint.attrib["damping"] = str(float(joint.attrib["damping"]) * joint_damping_scale)

    if actuator_gain_scale != 1.0 or actuator_bias_scale != 1.0:
        for act in root.iter("actuator"):
            for child in list(act):
                if actuator_gain_scale != 1.0 and "gainprm" in child.attrib:
                    gp = _float_list(child.attrib["gainprm"])
                    gp = [v * actuator_gain_scale for v in gp]
                    child.attrib["gainprm"] = _format_floats(gp)
                if actuator_bias_scale != 1.0 and "biasprm" in child.attrib:
                    bp = _float_list(child.attrib["biasprm"])
                    bp = [v * actuator_bias_scale for v in bp]
                    child.attrib["biasprm"] = _format_floats(bp)

    out_name = f"{pathlib.Path(base_xml_path).stem}__{run_name}.xml"
    out_path = os.path.join(out_dir, out_name)
    tree.write(out_path)
    return out_path


def make_mujoco_env(
    env_id: str,
    *,
    xml_out_dir: str,
    run_name: str,
    body_mass_scale: float,
    geom_friction_scale: float,
    joint_damping_scale: float,
    actuator_gain_scale: float,
    actuator_bias_scale: float,
    xml_path_override: Optional[str] = None,
    render_mode: Optional[str] = None,
):
    base_xml = xml_path_override or locate_mujoco_xml(env_id)
    xml_path = perturb_mujoco_xml(
        base_xml,
        xml_out_dir,
        run_name=run_name,
        body_mass_scale=body_mass_scale,
        geom_friction_scale=geom_friction_scale,
        joint_damping_scale=joint_damping_scale,
        actuator_gain_scale=actuator_gain_scale,
        actuator_bias_scale=actuator_bias_scale,
    )
    if render_mode:
        return gym.make(env_id, xml_file=xml_path, render_mode=render_mode)
    return gym.make(env_id, xml_file=xml_path)


__all__ = [
    "locate_mujoco_xml",
    "make_mujoco_env",
    "perturb_mujoco_xml",
]
