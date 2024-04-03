

# Original file: ./hf_GPT2___60.0/hf_GPT2___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/vh/cvh6pjoizkigx4e4rf7lezkkoaqevruw2u2j4hnifch2fimofjgz.py
# Source Nodes: [full, full_1, softmax, truediv, where], Original ATen: [aten._softmax, aten.div, aten.full, aten.where]
# full => full_default
# full_1 => full_default_1
# softmax => amax, convert_element_type_3, convert_element_type_4, div_1, exp, sub_1, sum_1
# truediv => div
# where => where
triton_per_fused__softmax_div_full_where_3 = async_compile.triton('triton_per_fused__softmax_div_full_where_3', '''
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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_full_where_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_div_full_where_3(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask, other=0.0).to(tl.float32)
    tmp2 = 8.0
    tmp3 = tmp1 / tmp2
    tmp4 = -3.3895313892515355e+38
    tmp5 = tl.where(tmp0, tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp9, 0))
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp12 / tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp18, rmask)
''')
