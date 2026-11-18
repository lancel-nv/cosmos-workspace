from importlib.metadata import version
import sys
from types import ModuleType


def enable_legacy_physx_imports():
    """Bridge relocated PhysX imports without substituting physics or measured state."""
    if int(version("isaacsim").split(".")[0]) < 6:
        return
    import omni.physics.tensors as tensors
    import omni.physics.tensors.api as api

    for name in ("SimulationView", "ArticulationView", "RigidBodyView", "create_simulation_view"):
        if not hasattr(api, name):
            raise RuntimeError(f"Isaac Sim PhysX tensor API is missing {name}")
    legacy = ModuleType("omni.physics.tensors.impl")
    legacy.__path__ = []
    legacy.api = api
    sys.modules.setdefault("omni.physics.tensors.impl", legacy)
    sys.modules.setdefault("omni.physics.tensors.impl.api", api)
    tensors.impl = sys.modules["omni.physics.tensors.impl"]
