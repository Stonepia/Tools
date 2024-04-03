

# Original file: ./mobilenet_v3_large___60.0/mobilenet_v3_large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ps/cpsnz3tc6ud6r42rsuljs24y5kzxmhyonc765falu5bp7jz24it6.py
# Source Nodes: [getattr_getattr_l__mod___features___5___block___2___scale_activation, mul_1], Original ATen: [aten.hardsigmoid, aten.mul]
# getattr_getattr_l__mod___features___5___block___2___scale_activation => add_32, clamp_max_2, clamp_min_2, convert_element_type_46, convert_element_type_47, div_2
# mul_1 => mul_44
triton_poi_fused_hardsigmoid_mul_3 = async_compile.triton('triton_poi_fused_hardsigmoid_mul_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_mul_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardsigmoid_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 120
    x2 = (xindex // 94080)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 3.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 6.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')
