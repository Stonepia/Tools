

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/i2/ci2s2mtlrux7ev3vqsxxi36hlffwbkanp266bg63re4ymcdme6ye.py
# Source Nodes: [l__mod___cell_4_comb_iter_1_left_act_1, pad_23], Original ATen: [aten.constant_pad_nd, aten.relu]
# l__mod___cell_4_comb_iter_1_left_act_1 => relu_89
# pad_23 => constant_pad_nd_25
triton_poi_fused_constant_pad_nd_relu_47 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_47', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_47', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12780288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 18576) % 43
    x1 = (xindex // 432) % 43
    x3 = (xindex // 798768)
    x4 = xindex % 18576
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (18144*x2) + (762048*x3)), tmp5 & xmask, other=0.0)
    tmp7 = triton_helpers.maximum(0, tmp6)
    tmp8 = tl.where(tmp5, tmp7, 0.0)
    tl.store(out_ptr0 + (x5), tmp8, xmask)
''')
