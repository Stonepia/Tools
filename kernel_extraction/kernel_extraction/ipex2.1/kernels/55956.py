

# Original file: ./cait_m36_384___60.0/cait_m36_384___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/em/cemrd3pjnmkmmjwzxdtfijodzkwq2gulzwl2czvwzsbru7xde55i.py
# Source Nodes: [add_73, add_74, l__self___blocks_token_only_0_attn_proj_drop, l__self___blocks_token_only_0_mlp_drop2, mul_108, mul_109], Original ATen: [aten.add, aten.clone, aten.mul]
# add_73 => add_327
# add_74 => add_331
# l__self___blocks_token_only_0_attn_proj_drop => clone_289
# l__self___blocks_token_only_0_mlp_drop2 => clone_291
# mul_108 => mul_364
# mul_109 => mul_370
triton_poi_fused_add_clone_mul_23 = async_compile.triton('triton_poi_fused_add_clone_mul_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_mul_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_mul_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    tmp7 = tl.load(in_ptr4 + (x0), xmask).to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp5 + tmp9
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')
