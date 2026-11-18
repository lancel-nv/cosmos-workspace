from __future__ import annotations

import numpy as np


def scene_resources(scene_manager):
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdLux

    stage = get_current_stage()
    room_root = stage.GetPrimAtPath(f"{scene_manager.env_roots[0]}/Rooms")
    if not room_root.IsValid():
        raise RuntimeError("Missing push_T room root")
    rooms = [str(instance.GetPath()) for category in room_root.GetChildren() for instance in category.GetChildren()]
    expected_room = scene_manager.config["env0"]["Room"]["prim_path"]
    if rooms != [expected_room]:
        raise RuntimeError(f"Stale or missing rooms after reset: {rooms}; expected {expected_room}")
    lights = [
        {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "intensity": prim.GetAttribute("inputs:intensity").Get(),
        }
        for prim in stage.Traverse()
        if prim.HasAPI(UsdLux.LightAPI)
    ]
    if sorted((light["type"], light["intensity"]) for light in lights) != [
        ("DomeLight", 1000.0),
        ("RectLight", 15000.0),
    ]:
        raise RuntimeError(f"Unexpected lights for the fixed push_T scene: {lights}")
    return {"room_count": len(rooms), "lights": lights}


def head_exposure(frame):
    pixels = np.asarray(frame)
    if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"Unexpected head RGB shape/dtype: {pixels.shape}, {pixels.dtype}")
    tabletop = pixels[120:360, 128:512]
    return {
        "table_mean_rgb": float(tabletop.mean()),
        "table_white_fraction": float((tabletop.min(axis=-1) >= 245).mean()),
    }


def check_head_exposure(frame):
    metrics = head_exposure(frame)
    if metrics["table_white_fraction"] > 0.10:
        raise RuntimeError(f"Head-view exposure guard failed; retain partial data for inspection: {metrics}")
    return metrics
