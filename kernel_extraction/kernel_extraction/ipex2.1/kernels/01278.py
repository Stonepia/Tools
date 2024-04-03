

# Original file: ./PLBartForConditionalGeneration___60.0/PLBartForConditionalGeneration___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/kq/ckqfvj5hxykbnhuipwqwu3lv3r4r6plfbbdoyvrcbrmfgcdpx73a.py
# Source Nodes: [clone, clone_1, eq, masked_fill_, ne, setitem, setitem_1, sum_1], Original ATen: [aten.clone, aten.copy, aten.eq, aten.masked_fill, aten.ne, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sum]
# clone => clone
# clone_1 => clone_1
# eq => eq
# masked_fill_ => full_default, where
# ne => ne
# setitem => copy, slice_9, slice_scatter, slice_scatter_1
# setitem_1 => copy_1, select_scatter, slice_14, slice_scatter_2
# sum_1 => sum_1
triton_per_fused_clone_copy_eq_masked_fill_ne_select_scatter_slice_slice_scatter_sum_0 = async_compile.triton('triton_per_fused_clone_copy_eq_masked_fill_ne_select_scatter_slice_slice_scatter_sum_0', '''
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
    size_hints=[4, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_copy_eq_masked_fill_ne_select_scatter_slice_slice_scatter_sum_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_clone_copy_eq_masked_fill_ne_select_scatter_slice_slice_scatter_sum_0(in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp5 = tmp4 != tmp3
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = r1
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = tmp11 == tmp12
    tmp14 = tmp10 - tmp3
    tmp15 = tl.where(tmp14 < 0, tmp14 + 1024, tmp14)
    # tl.device_assert((0 <= tmp15) & (tmp15 < 1024), "index out of bounds: 0 <= tmp15 < 1024")
    tmp16 = tl.load(in_ptr0 + (tmp15 + (1024*x0)), xmask)
    tmp17 = tmp16 == tmp1
    tmp18 = tl.where(tmp17, tmp3, tmp16)
    tmp19 = tmp11 >= tmp3
    tmp20 = tl.load(in_ptr0 + ((-1) + r1 + (1024*x0)), rmask & tmp19 & xmask, other=0.0)
    tmp21 = tmp20 == tmp1
    tmp22 = tl.where(tmp21, tmp3, tmp20)
    tmp23 = tl.where(tmp19, tmp22, 0)
    tmp24 = tl.where(tmp19, tmp23, tmp4)
    tmp25 = tl.where(tmp13, tmp18, tmp24)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')
