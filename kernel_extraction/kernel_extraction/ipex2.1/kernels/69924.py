

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/td/ctdbje5mz6vqoiioahjmy2r7oqwlmgues2676unzwqzzgllxoz2v.py
# Source Nodes: [l__mod___encoder_b_encoder], Original ATen: [aten._softmax, aten.add, aten.logical_not, aten.masked_fill, aten.ones, aten.tril, aten.zeros_like]
# l__mod___encoder_b_encoder => add_92, amax_12, div_12, exp_12, full_default, full_default_1, full_default_2, le, logical_and, logical_not, sub_39, sub_40, sum_13, where
triton_per_fused__softmax_add_logical_not_masked_fill_ones_tril_zeros_like_19 = async_compile.triton('triton_per_fused__softmax_add_logical_not_masked_fill_ones_tril_zeros_like_19', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_logical_not_masked_fill_ones_tril_zeros_like_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_logical_not_masked_fill_ones_tril_zeros_like_19(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 19712
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 77
    tmp0 = tl.load(in_ptr0 + (r2 + (77*x3)), rmask & xmask, other=0.0)
    tmp1 = r2 + ((-1)*x0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1, 1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = tmp5 == 0
    tmp7 = float("-inf")
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp0 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, float("-inf"))
    tmp14 = triton_helpers.max2(tmp13, 1)[:, None]
    tmp15 = tmp10 - tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp16 / tmp20
    tl.store(out_ptr2 + (r2 + (77*x3)), tmp21, rmask & xmask)
''')
