

# Original file: ./ese_vovnet19b_dw___60.0/ese_vovnet19b_dw___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/sm/csmhybi5v63wjmspeclkuonjmolpmltjnxzguz3xc3kypkstcpzb.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_gate, mul_1], Original ATen: [aten.hardsigmoid, aten.mul]
# getattr_getattr_l__mod___stages___1___blocks___0___attn_gate => add_27, clamp_max_1, clamp_min_1, convert_element_type_41, convert_element_type_42, div_1
# mul_1 => mul_40
triton_poi_fused_hardsigmoid_mul_9 = async_compile.triton('triton_poi_fused_hardsigmoid_mul_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_mul_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardsigmoid_mul_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 84934656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 663552)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp0 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')
