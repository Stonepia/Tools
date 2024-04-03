

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/2b/c2b4fmxm4oikshkt55nn3be74cxxagkihldxgkfw3snbxm6m3673.py
# Source Nodes: [l__self___encoder_a_encoder], Original ATen: [aten.clone]
# l__self___encoder_a_encoder => clone_1
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1228800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768) % 32
    x2 = (xindex // 24576)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*x2) + (38400*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')