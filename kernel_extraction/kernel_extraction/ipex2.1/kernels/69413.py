

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/bs/cbse44cm6cddv7goqcm7nkiy2av3d66p4xlplbfguew2tkbfk4ub.py
# Source Nodes: [add_26, add_27, mul_26, mul_27, sub_13], Original ATen: [aten.add, aten.mul, aten.sub]
# add_26 => add_93
# add_27 => add_97
# mul_26 => mul_119
# mul_27 => mul_125
# sub_13 => sub_40
triton_poi_fused_add_mul_sub_44 = async_compile.triton('triton_poi_fused_add_mul_sub_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_sub_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp10, xmask)
''')
