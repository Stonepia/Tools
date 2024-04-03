

# Original file: ./fastNLP_Bert___60.0/fastNLP_Bert___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/6o/c6ok3flt4pjwidao777uwc7zsh4icshbflth7z6b7zq7gv65yd6q.py
# Source Nodes: [eq, getitem, masked_fill, max_1, ne, sum_1, sum_2], Original ATen: [aten.eq, aten.index, aten.masked_fill, aten.max, aten.ne, aten.sum]
# eq => eq
# getitem => index
# masked_fill => full_default, where
# max_1 => max_1
# ne => ne
# sum_1 => sum_1
# sum_2 => sum_2
triton_per_fused_eq_index_masked_fill_max_ne_sum_0 = async_compile.triton('triton_per_fused_eq_index_masked_fill_max_ne_sum_0', '''
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
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*i1', 3: '*i64', 4: '*i64', 5: '*i64', 6: '*i64', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_eq_index_masked_fill_max_ne_sum_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_eq_index_masked_fill_max_ne_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 473
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], False, tl.int1)
    tmp4 = tmp2 == tmp3
    tmp5 = tl.where(tmp0 < 0, tmp0 + 2869, tmp0)
    # tl.device_assert(((0 <= tmp5) & (tmp5 < 2869)) | ~rmask, "index out of bounds: 0 <= tmp5 < 2869")
    tmp6 = tl.load(in_ptr1 + (tmp5), rmask)
    tmp7 = tl.where(tmp4, tmp1, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp2.to(tl.int64)
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp2, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp7, rmask)
    tl.store(out_ptr4 + (tl.full([1], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr3 + (tl.full([1], 0, tl.int32)), tmp16, None)
''')
