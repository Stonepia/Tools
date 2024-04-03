

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/o4/co4anruwte4vhm6racqi3gu7vbe7xrxnl6v4yasi4i6id5y76aq2.py
# Source Nodes: [l__self___cell_8_comb_iter_1_left_act_1, pad_35], Original ATen: [aten.constant_pad_nd, aten.relu]
# l__self___cell_8_comb_iter_1_left_act_1 => relu_146
# pad_35 => constant_pad_nd_38
triton_poi_fused_constant_pad_nd_relu_61 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_61', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_61', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_61(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6096384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, xmask)
''')
