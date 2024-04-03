

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/6j/c6jjpe2novn6dchfowntxmbn5hp5cckad2px665b2ac6dslbewjr.py
# Source Nodes: [l__mod___cell_4_comb_iter_1_left_act_1, pad_26], Original ATen: [aten.constant_pad_nd, aten.relu]
# l__mod___cell_4_comb_iter_1_left_act_1 => relu_89
# pad_26 => constant_pad_nd_28
triton_poi_fused_constant_pad_nd_relu_49 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_49', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_49', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_49(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12192768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, xmask)
''')
