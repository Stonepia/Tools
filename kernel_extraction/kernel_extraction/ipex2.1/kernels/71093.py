

# Original file: ./timm_nfnet___60.0/timm_nfnet___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/eo/ceo63zwtwm7qoxzbb4xuafe4p4m4pj23fw5rjyeqlrybzfoksh6h.py
# Source Nodes: [add_4, gelu_23, mul_44, mul_45, mul_46, mul_47, mul__27, mul__28], Original ATen: [aten.add, aten.gelu, aten.mul]
# add_4 => add_54
# gelu_23 => add_55, convert_element_type_100, convert_element_type_101, erf_23, mul_198, mul_199, mul_200
# mul_44 => mul_194
# mul_45 => mul_195
# mul_46 => mul_197
# mul_47 => mul_202
# mul__27 => mul_196
# mul__28 => mul_201
triton_poi_fused_add_gelu_mul_20 = async_compile.triton('triton_poi_fused_add_gelu_mul_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_gelu_mul_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28311552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 221184)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 * tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = 0.7071067811865476
    tmp15 = tmp11 * tmp14
    tmp16 = libdevice.erf(tmp15)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = tmp13 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1.7015043497085571
    tmp22 = tmp20 * tmp21
    tmp23 = 0.9622504486493761
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x3), tmp24, None)
''')
