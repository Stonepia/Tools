

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/tm/ctmdijw55k4a632dnbgmz6tgcgd3ene3b3e4rc5gvooakc2bcgxf.py
# Source Nodes: [l__self___cls_attn_blocks_0_mlp_drop2, mul_100], Original ATen: [aten.clone, aten.mul]
# l__self___cls_attn_blocks_0_mlp_drop2 => clone_245
# mul_100 => mul_487
triton_poi_fused_clone_mul_35 = async_compile.triton('triton_poi_fused_clone_mul_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_mul_35(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0 + (602880*x1)), tmp3, xmask)
''')
