

# Original file: ./XGLMForCausalLM__55_backward_294.40/XGLMForCausalLM__55_backward_294.40_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/zy/czy7x2i2qx72a3qgwp6pkoehqki5bsyy76f3othpckquyc64smbd.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.masked_fill, aten.native_dropout_backward, aten.where]

triton_per_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_where_8 = async_compile.triton('triton_per_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_where_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*i1', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_where_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_where_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp6 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask)
    tmp13 = tl.load(in_ptr4 + (r1 + (128*x0)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp14 = tmp6 * tmp11
    tmp15 = tmp7 - tmp14
    tmp16 = 2.0
    tmp17 = tmp15 / tmp16
    tmp18 = tl.where(tmp13, tmp17, tmp15)
    tmp19 = 0.0
    tmp20 = tl.where(tmp12, tmp19, tmp18)
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp20, rmask)
''')
