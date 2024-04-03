

# Original file: ./eca_botnext26ts_256___60.0/eca_botnext26ts_256___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/rq/crq54zd7slxl5423wxntw5vsvz2tmuvoblnslpoocenr5tf562kf.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_act, mul_4], Original ATen: [aten.mul, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_act => convert_element_type_90, mul_77, sigmoid_20
# mul_4 => mul_78
triton_poi_fused_mul_silu_21 = async_compile.triton('triton_poi_fused_mul_silu_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_silu_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 65536)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tl.store(out_ptr0 + (x3), tmp6, None)
''')
