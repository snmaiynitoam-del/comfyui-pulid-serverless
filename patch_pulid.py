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

if patched:
    with open(PULID_FILE, "w") as f:
        f.write(content)
    print(f"Patches applied to {PULID_FILE}")
else:
    print("No patches needed or file already patched")
