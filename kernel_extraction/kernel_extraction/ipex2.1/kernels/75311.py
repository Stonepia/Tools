

# Original file: ./XGLMForCausalLM__94_forward_279.26/XGLMForCausalLM__94_forward_279.26.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/67/c67cuftmj7q3jmd4kyeqcbcsuu5esfmtfbsvtqwh6cdbclpoqcqd.py
# Source Nodes: [clone, cross_entropy, new_zeros, setitem, setitem_1], Original ATen: [aten.clone, aten.copy, aten.new_zeros, aten.nll_loss_forward, aten.select_scatter, aten.slice, aten.slice_scatter]
# clone => clone
# cross_entropy => convert_element_type_3, div, full_default_1, ne, neg, sum_2, sum_3, where_1
# new_zeros => full
# setitem => copy, slice_3, slice_scatter, slice_scatter_1
# setitem_1 => copy_1, select_scatter, slice_9, slice_scatter_2
triton_per_fused_clone_copy_new_zeros_nll_loss_forward_select_scatter_slice_slice_scatter_3 = async_compile.triton('triton_per_fused_clone_copy_new_zeros_nll_loss_forward_select_scatter_slice_slice_scatter_3', '''
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
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_copy_new_zeros_nll_loss_forward_select_scatter_slice_slice_scatter_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_clone_copy_new_zeros_nll_loss_forward_select_scatter_slice_slice_scatter_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex % 128
    r2 = rindex
    tmp0 = r0
    tmp1 = tl.full([1], 127, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 127, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r2, [RBLOCK])), rmask & tmp4, other=0.0)
    tmp6 = tl.where(tmp4, tmp5, 0)
    tmp7 = tl.full([1], 0, tl.int64)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tl.where(tmp2, tmp9, tmp8)
    tmp11 = tl.full([1], -100, tl.int64)
    tmp12 = tmp10 != tmp11
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.where(tmp12, tmp10, tmp7)
    tmp19 = tl.where(tmp18 < 0, tmp18 + 256008, tmp18)
    # tl.device_assert((0 <= tmp19) & (tmp19 < 256008), "index out of bounds: 0 <= tmp19 < 256008")
    tmp20 = tl.load(in_ptr1 + (tmp19 + (256008*r2)), rmask, other=0.0)
    tmp21 = -tmp20
    tmp22 = 0.0
    tmp23 = tl.where(tmp12, tmp21, tmp22)
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp17.to(tl.float32)
    tmp29 = tmp27 / tmp28
    tl.store(out_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), tmp10, rmask)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp28, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp29, None)
''')
