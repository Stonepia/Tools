

# Original file: ./mixnet_l___60.0/mixnet_l___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/u5/cu5o4m2eytdhbs7csf4p77bflmfmvnkbfmqbijgcdkuztgbhjfw6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___bn2_act, mul_4], Original ATen: [aten.mul, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___bn2_act => mul_86, sigmoid_17
# mul_4 => mul_88
triton_poi_fused_mul_silu_37 = async_compile.triton('triton_poi_fused_mul_silu_37', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_37', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_silu_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8429568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 336
    x2 = (xindex // 65856)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (336*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')