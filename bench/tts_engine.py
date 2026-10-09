from __future__ import annotations

from bench.common import MODELS_DIR


def load_kokoro():
    """Return (Kokoro, provider). Uses CUDA when onnxruntime-gpu can load it, else CPU."""
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
    providers = [p for p in wanted if p in ort.get_available_providers()]
    session = ort.InferenceSession(str(MODELS_DIR / "kokoro" / "kokoro-v1.0.onnx"), providers=providers)
    provider = session.get_providers()[0]
    if provider != "CUDAExecutionProvider":
        print(f"WARNING: Kokoro running on {provider}, not CUDA")
    return Kokoro.from_session(session, str(MODELS_DIR / "kokoro" / "voices-v1.0.bin")), provider
