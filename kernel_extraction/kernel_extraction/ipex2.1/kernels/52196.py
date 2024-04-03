

# Original file: ./mobilenetv3_large_100___60.0/mobilenetv3_large_100___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/77/c773mshcv5bctefjavdwharxk4vzvjmmdwxl2vdbbsgmagmpifkn.py
# Source Nodes: [l__self___act2], Original ATen: [aten.hardswish]
# l__self___act2 => add_130, clamp_max_28, clamp_min_28, convert_element_type_275, convert_element_type_276, div_28, mul_166
triton_poi_fused_hardswish_26 = async_compile.triton('triton_poi_fused_hardswish_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardswish_26(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 3.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 6.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8 / tmp6
    tmp10 = tmp9.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''')
