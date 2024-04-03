

# Original file: ./moondream___60.0/moondream___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/3d/c3ddsxzgahpsv3folil33cq4s2vg67zkvq7go4y3mn2sh4zi6yqo.py
# Source Nodes: [add_1, add_2, mul, mul_1, mul_2, mul_3], Original ATen: [aten.add, aten.mul]
# add_1 => add_3
# add_2 => add_4
# mul => mul_2
# mul_1 => mul_3
# mul_2 => mul_4
# mul_3 => mul_5
triton_poi_fused_add_mul_3 = async_compile.triton('triton_poi_fused_add_mul_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 512
    x2 = (xindex // 16384)
    x3 = xindex % 16384
    x4 = xindex
    x5 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (2048*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x0 + (64*x2) + (2048*x1)), None).to(tl.float32)
    tmp9 = tl.load(in_ptr5 + (x4), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp7 * tmp1
    tmp10 = tmp9 * tmp4
    tmp11 = tmp8 + tmp10
    tl.store(out_ptr0 + (x0 + (64*x5)), tmp6, None)
    tl.store(out_ptr1 + (x0 + (64*x5)), tmp11, None)
''')
