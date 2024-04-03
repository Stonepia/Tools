

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/g7/cg7ukguinj6oeb5kqrpjikogqgv2uyqeoxletzp52jlkzp4e3qyf.py
# Source Nodes: [add_74, l__mod___blocks_23_mlp_drop2, mul_98], Original ATen: [aten.add, aten.clone, aten.mul]
# add_74 => add_349
# l__mod___blocks_23_mlp_drop2 => clone_264
# mul_98 => mul_476
triton_poi_fused_add_clone_mul_22 = async_compile.triton('triton_poi_fused_add_clone_mul_22', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_mul_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_mul_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 602112)
    x4 = xindex % 602112
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4 + (602880*x2)), tmp4, None)
''')
