

# Original file: ./dm_nfnet_f0___60.0/dm_nfnet_f0___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/4e/c4eqqgu4uyer4mmtsrhwwfjeg5hzovou4ixn53guc5kikctw2dws.py
# Source Nodes: [add_2, gelu_15, mul_27, mul_28, mul_29, mul_30, mul__17, mul__18], Original ATen: [aten.add, aten.gelu, aten.mul]
# add_2 => add_35
# gelu_15 => add_36, convert_element_type_79, convert_element_type_80, erf_15, mul_129, mul_130, mul_131
# mul_27 => mul_125
# mul_28 => mul_126
# mul_29 => mul_128
# mul_30 => mul_133
# mul__17 => mul_127
# mul__18 => mul_132
triton_poi_fused_add_gelu_mul_13 = async_compile.triton('triton_poi_fused_add_gelu_mul_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_gelu_mul_13(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 524288)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = -1.4702593088150024
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
    tl.store(in_out_ptr0 + (x3), tmp24, None)
''')
