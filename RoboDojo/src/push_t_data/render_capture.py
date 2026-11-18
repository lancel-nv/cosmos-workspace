from __future__ import annotations

RENDER_CONFIG = {
    "enable_dl_denoiser": False,
    "antialiasing_mode": None,
    "carb_settings": {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 16,
        "/rtx/pathtracing/totalSpp": 16,
        "/rtx/resetPtAccumOnAnimTimeChange": True,
        "/rtx-transient/resetPtAccumOnlyWhenExternalFrameCounterChanges": False,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
        "/rtx/pathtracing/optixDenoiser/temporalMode/enabled": False,
        "/rtx/pathtracing/optixDenoiser/blendFactor": 0.0,
        "/rtx/pathtracing/cached/enabled": False,
        "/rtx/pathtracing/lightcache/cached/enabled": False,
    },
}

CAPTURE_SEMANTICS = "untiled_pathtrace16_spatial_optix_replicator_zero_time_v1"


def capture_metadata():
    from copy import deepcopy

    return {
        "semantics": CAPTURE_SEMANTICS,
        "config": deepcopy(RENDER_CONFIG),
        "use_tiled": False,
        "bootstrap_reset_discarded_per_process": 1,
    }


def validate_capture_metadata(metadata):
    if metadata.get("render_capture") != capture_metadata():
        raise ValueError("Missing or incompatible synchronous capture settings")


def render_capture(env):
    import omni.replicator.core as rep

    physics_time = env.sim.sim.current_time
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False, wait_for_render=True)
    if env.sim.sim.current_time != physics_time:
        raise RuntimeError("Camera capture advanced physics")
