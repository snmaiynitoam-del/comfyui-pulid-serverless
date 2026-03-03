"""
Patches ComfyUI-PuLID-Flux for compatibility with latest ComfyUI.
Run after cloning the PuLID repo into custom_nodes.
"""
import re
import os

PULID_DIR = "/comfyui/custom_nodes/ComfyUI-PuLID-Flux"
PULID_FILE = os.path.join(PULID_DIR, "pulidflux.py")

if not os.path.exists(PULID_FILE):
    print(f"PuLID file not found at {PULID_FILE}, skipping patches")
    exit(0)

with open(PULID_FILE, "r") as f:
    content = f.read()

patched = False

# Fix 1: Add attn_mask parameter to forward_orig
old_sig = "def forward_orig(\n        self,\n        img: Tensor,\n        img_ids: Tensor,\n        txt: Tensor,\n        txt_ids: Tensor,\n        timesteps: Tensor,\n        y: Tensor,\n        guidance: Tensor = None,\n        control=None,\n        transformer_options={},"
new_sig = "def forward_orig(\n        self,\n        img: Tensor,\n        img_ids: Tensor,\n        txt: Tensor,\n        txt_ids: Tensor,\n        timesteps: Tensor,\n        y: Tensor,\n        guidance: Tensor = None,\n        control=None,\n        transformer_options={},\n        attn_mask: Tensor = None,"

if "attn_mask: Tensor = None," not in content and "def forward_orig(" in content:
    # Use regex for more flexible matching
    content = re.sub(
        r"(def forward_orig\([^)]*transformer_options\s*=\s*\{\})\s*,?\s*\)",
        r"\1,\n        attn_mask: Tensor = None,\n        **kwargs,\n    ) -> Tensor:",
        content,
        count=1
    )
    # If regex didn't match, try simple replacement
    if "attn_mask" not in content:
        content = content.replace(
            "transformer_options={},\n    ) -> Tensor:",
            "transformer_options={},\n        attn_mask: Tensor = None,\n        **kwargs,\n    ) -> Tensor:"
        )
    patched = True
    print("Fix 1: Added attn_mask parameter")

# Fix 2: Ensure PuLID CA layers are moved to correct device
# For double blocks
old_double = "ca = self.pulid_ca[ca_idx]"
new_double = "ca = self.pulid_ca[ca_idx].to(img.device, dtype=img.dtype)"
if old_double in content:
    content = content.replace(old_double, new_double)
    patched = True
    print("Fix 2: Fixed device placement for CA layers")

# Fix 3: Ensure embeddings are on correct device
old_emb = "node_data['embedding']"
new_emb = "node_data['embedding'].to(img.device, dtype=img.dtype)"
# Only replace in the CA call context, not everywhere
old_ca_call = "ca(node_data['embedding'], img)"
new_ca_call = "ca(node_data['embedding'].to(img.device, dtype=img.dtype), img)"
if old_ca_call in content:
    content = content.replace(old_ca_call, new_ca_call)
    patched = True
    print("Fix 3: Fixed embedding device placement")

# Also fix for real_img in single blocks
old_ca_single = "ca(node_data['embedding'], real_img)"
new_ca_single = "ca(node_data['embedding'].to(real_img.device, dtype=real_img.dtype), real_img)"
if old_ca_single in content:
    content = content.replace(old_ca_single, new_ca_single)
    print("Fix 3b: Fixed embedding device placement in single blocks")

# Fix 4: Local EVA-CLIP loading
if "clip_file_path" not in content and "load_eva_clip" in content:
    old_clip = "model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)"
    new_clip = """clip_file_path = os.path.join(folder_paths.models_dir, "clip", "EVA02_CLIP_L_336_psz14_s6B.pt")
        if os.path.exists(clip_file_path):
            model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', clip_file_path, force_custom_clip=True)
        else:
            model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)"""
    if old_clip in content:
        content = content.replace(old_clip, new_clip)
        patched = True
        print("Fix 4: Added local EVA-CLIP loading")

# Fix 5: Add model_rootpath to FaceRestoreHelper
if "model_rootpath" not in content and "FaceRestoreHelper" in content:
    old_face = "face_helper = FaceRestoreHelper("
    new_face = """FACEXLIB_DIR = os.path.join(folder_paths.models_dir, "facexlib")
        face_helper = FaceRestoreHelper(
            model_rootpath=FACEXLIB_DIR,"""
    # Only add if not already patched
    if "FACEXLIB_DIR" not in content:
        content = content.replace(old_face, new_face, 1)
        patched = True
        print("Fix 5: Added model_rootpath to FaceRestoreHelper")

# Fix 6: Fix InsightFace FaceAnalysis providers argument (newer insightface API change)
# Newer insightface versions don't accept 'providers' in __init__, must be passed separately
old_fa = """model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])"""
new_fa = """model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR)"""
if old_fa in content:
    content = content.replace(old_fa, new_fa)
    patched = True
    print("Fix 6: Removed providers kwarg from FaceAnalysis (newer insightface compat)")
else:
    # Try regex match for variations
    content, n = re.subn(
        r"FaceAnalysis\(name=\"antelopev2\",\s*root=INSIGHTFACE_DIR,\s*providers=\[[^\]]*\]\)",
        'FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR)',
        content
    )
    if n > 0:
        patched = True
        print("Fix 6: Removed providers kwarg from FaceAnalysis (regex)")

# Fix 7: Replace InsightFace loading with robust multi-path version + TensorRT stripping
# Root cause: InsightFace passes providers EXPLICITLY to onnxruntime.InferenceSession,
# so checking "if providers is None" never fires. TensorrtExecutionProvider silently fails
# to load ONNX models on RunPod, causing 'assert detection in self.models'.
# Fix: unconditionally strip TensorRT from ALL InferenceSession calls AND hide it from
# get_available_providers() so InsightFace never sees it.
old_fa_call = 'model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR)'
new_fa_call = '''import glob as _glob
        import onnxruntime as _ort
        # Debug: print ONNX runtime info
        print(f"[PuLID-Debug] onnxruntime version: {_ort.__version__}")
        print(f"[PuLID-Debug] Available providers (raw): {_ort.get_available_providers()}")
        print(f"[PuLID-Debug] INSIGHTFACE_DIR = {INSIGHTFACE_DIR}")
        _search_paths = [
            INSIGHTFACE_DIR,
            "/runpod-volume/insightface",
            os.path.join(folder_paths.models_dir, "insightface"),
        ]
        _actual_root = None
        for _try_root in _search_paths:
            _model_dir = os.path.join(_try_root, "models", "antelopev2")
            _onnx_files = _glob.glob(os.path.join(_model_dir, "*.onnx"))
            if _onnx_files:
                _actual_root = _try_root
                print(f"[PuLID-Debug] Using root: {_actual_root} ({len(_onnx_files)} .onnx files)")
                break
        if _actual_root is None:
            _actual_root = INSIGHTFACE_DIR

        # ====== TensorRT stripping (Fix 10) ======
        # Patch get_available_providers() to hide TensorRT entirely.
        # InsightFace uses this to build its default providers list.
        _orig_get_providers = _ort.get_available_providers
        def _patched_get_providers():
            return [p for p in _orig_get_providers() if 'Tensorrt' not in p and 'TensorRT' not in p]
        _ort.get_available_providers = _patched_get_providers

        # Patch InferenceSession.__init__ to UNCONDITIONALLY strip TensorRT
        # from any providers list (InsightFace always passes providers explicitly).
        _orig_session_init = _ort.InferenceSession.__init__
        def _patched_session_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            if providers is not None:
                _filtered = []
                _filtered_opts = []
                for _i, _p in enumerate(providers):
                    _pname = _p[0] if isinstance(_p, tuple) else _p
                    if 'Tensorrt' not in _pname and 'TensorRT' not in _pname:
                        _filtered.append(_p)
                        if provider_options is not None and _i < len(provider_options):
                            _filtered_opts.append(provider_options[_i])
                    else:
                        print(f"[PuLID-Debug] Stripped {_pname} from session providers")
                providers = _filtered if _filtered else ['CUDAExecutionProvider', 'CPUExecutionProvider']
                provider_options = _filtered_opts if _filtered_opts else None
                if provider_options is not None and len(provider_options) == 0:
                    provider_options = None
            print(f"[PuLID-Debug] InferenceSession providers: {providers}")
            _orig_session_init(self, path_or_bytes, sess_options, providers=providers, provider_options=provider_options, **kwargs)
        _ort.InferenceSession.__init__ = _patched_session_init

        print(f"[PuLID-Debug] Effective providers after patch: {_ort.get_available_providers()}")
        try:
            model = FaceAnalysis(name="antelopev2", root=_actual_root)
            print(f"[PuLID-Debug] FaceAnalysis loaded successfully! Models: {list(model.models.keys())}")
        except Exception as _e:
            print(f"[PuLID-Debug] FaceAnalysis failed: {_e}")
            raise
        finally:
            # Restore originals so other code is unaffected
            _ort.InferenceSession.__init__ = _orig_session_init
            _ort.get_available_providers = _orig_get_providers'''

if old_fa_call in content:
    content = content.replace(old_fa_call, new_fa_call)
    patched = True
    print("Fix 7+10: Robust InsightFace loading with TensorRT stripping")
else:
    # If Fix 6 already ran, the line should match. If not, try regex
    content, n = re.subn(
        r'model\s*=\s*FaceAnalysis\(name="antelopev2",\s*root=INSIGHTFACE_DIR\)',
        new_fa_call,
        content,
        count=1
    )
    if n > 0:
        patched = True
        print("Fix 7+10: Robust InsightFace loading with TensorRT stripping (regex)")

if patched:
    with open(PULID_FILE, "w") as f:
        f.write(content)
    print(f"Patches applied to {PULID_FILE}")
else:
    print("No patches needed or file already patched")
