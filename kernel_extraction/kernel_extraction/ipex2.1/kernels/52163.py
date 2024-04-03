

# Original file: ./mobilenetv3_large_100___60.0/mobilenetv3_large_100___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/rm/crm6w4rm7bjypysnzutn4n6ew5n6dgnl3bno7wihrlospcryzxn3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___bn2_act, getattr_getattr_l__mod___blocks___5_____0___se_gate, mul_5], Original ATen: [aten.hardsigmoid, aten.hardswish, aten.mul]
# getattr_getattr_l__mod___blocks___5_____0___bn2_act => add_103, clamp_max_19, clamp_min_19, convert_element_type_153, div_19, mul_133
# getattr_getattr_l__mod___blocks___5_____0___se_gate => add_104, clamp_max_20, clamp_min_20, convert_element_type_154, convert_element_type_155, div_20
# mul_5 => mul_134
triton_poi_fused_hardsigmoid_hardswish_mul_20 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4214784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 32928)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr1 + (x0 + (672*x2)), None, eviction_policy='evict_last').to(tl.float32)
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