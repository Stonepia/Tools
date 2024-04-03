

# Original file: ./hf_DistilBert___60.0/hf_DistilBert___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/lm/clmpcuwj4pzdp2jbnkukd7flbfz4lfbxixulq5zkgkosqf6d457y.py
# Source Nodes: [masked_fill, softmax], Original ATen: [aten._softmax, aten.masked_fill]
# masked_fill => full_default, where
# softmax => amax, convert_element_type_3, convert_element_type_4, div_1, exp, sub_1, sum_1
triton_per_fused__softmax_masked_fill_3 = async_compile.triton('triton_per_fused__softmax_masked_fill_3', '''
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_masked_fill_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_masked_fill_3(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp0 = 1.0
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = -65504.0
    tmp5 = tl.where(tmp2, tmp4, tmp3)
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
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp18, rmask)
''')
