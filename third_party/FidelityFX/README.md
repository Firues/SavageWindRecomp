# AMD FidelityFX SDK 1.1.4 (subset)

From <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/v1.1.4>, MIT (`LICENSE.txt`). Savage Wind uses
FSR 3's frame interpolation and optical flow (`runtime/nvn/vulkan/framegen.cpp`) and its upscaler
(`runtime/nvn/vulkan/fsr3.cpp`) on the Vulkan backend, driven directly, without the SDK's frame interpolation
swapchain or its ffx-api DLL layer.

What is here:

- `sdk/include/FidelityFX/host`: the SDK's public headers; `gpu`: the shader headers of frame interpolation, optical
  flow, the FSR 3 upscaler, SPD (which optical flow and the upscaler include) and FSR 1 (the upscaler's RCAS).
- `sdk/src/components/frameinterpolation`, `opticalflow`, `fsr3upscaler`: the three effects.
- `sdk/src/backends/vk/ffx_vk.cpp` and its `shaders/frameinterpolation`, `shaders/opticalflow`,
  `shaders/fsr3upscaler` (GLSL).
- `sdk/src/shared`: the SDK's assert, message and object-management helpers.
- `sdk/src/backends/shared/ffx_shader_blobs.h`: the declaration of the blob lookup, which `framegen.cpp` implements.
- `spirv/ffx_shaders_vk.h`: the shaders compiled to SPIR-V with their bindings, by `tools/ffx_shaders.py`
  (glslangValidator) in place of the SDK's Windows-only FidelityFX_SC.exe. Only the permutations Savage Wind uses.
  Regenerate after changing the GLSL:
  `python3 tools/ffx_shaders.py third_party/FidelityFX/sdk third_party/FidelityFX/spirv/ffx_shaders_vk.h`.
- `spirv/ffx_fsr3upscaler_vk.h`: the same for the FSR 3 upscaler, one permutation per pass (SDR input,
  low-resolution motion vectors without jitter, depth not inverted):
  `python3 tools/ffx_shaders.py third_party/FidelityFX/sdk third_party/FidelityFX/spirv/ffx_fsr3upscaler_vk.h
  glslangValidator --effects fu`.
- `ffx_compat.h`: force-included into the SDK's sources so they build with clang and gcc (MinGW-w64, Linux).

Changes to the SDK's files (marked "Savage Wind" in them):

- `sdk/src/backends/vk/ffx_vk.cpp`: the arrays the backend lays out in its scratch memory are aligned to 64 bytes
  instead of 4, and start 64-byte aligned after its context. With 4, clang's aligned SSE stores into them (`movdqa`)
  faulted; MSVC emits unaligned ones.
- `sdk/include/FidelityFX/host/ffx_types.h`: the contexts' default size is doubled outside Windows, where `wchar_t`
  (their resources' debug names) is 4 bytes instead of 2.
