

# Original file: ./fbnetv3_b___60.0/fbnetv3_b___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/2p/c2p26rltqrglzxsyrnjk7hwff33yklnladevnf44acswbqscolni.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___bn2_act, getattr_getattr_l__mod___blocks___2_____0___se_gate, mul], Original ATen: [aten.hardsigmoid, aten.hardswish, aten.mul]
# getattr_getattr_l__mod___blocks___2_____0___bn2_act => add_55, clamp_max_12, clamp_min_12, convert_element_type_82, div_12, mul_69
# getattr_getattr_l__mod___blocks___2_____0___se_gate => add_57, clamp_max_14, clamp_min_14, convert_element_type_85, convert_element_type_86, div_14
# mul => mul_71
triton_poi_fused_hardsigmoid_hardswish_mul_10 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 120
    x2 = (xindex // 122880)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr1 + (x0 + (120*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp9 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')
