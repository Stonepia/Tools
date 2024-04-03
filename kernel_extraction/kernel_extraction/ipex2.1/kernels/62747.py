

# Original file: ./DebertaForQuestionAnswering__0_forward_133.0/DebertaForQuestionAnswering__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/ap/caplrxzhgjdrbxp3b4seirrf7xccyo3yp23k6mqbmerluuyvrqnn.py
# Source Nodes: [add_98, clamp, clamp_1, cross_entropy, cross_entropy_1, trampoline_autograd_apply, truediv_37], Original ATen: [aten.add, aten.clamp, aten.div, aten.masked_fill, aten.nll_loss_forward]
# add_98 => add_111
# clamp => clamp_max, clamp_min
# clamp_1 => clamp_max_1, clamp_min_1
# cross_entropy => convert_element_type_211, div_49, ne, neg, sum_14, sum_15, where_62
# cross_entropy_1 => convert_element_type_214, div_50, ne_3, neg_1, sum_17, sum_18, where_64
# trampoline_autograd_apply => full_default_1
# truediv_37 => div_51
triton_per_fused_add_clamp_div_masked_fill_nll_loss_forward_11 = async_compile.triton('triton_per_fused_add_clamp_div_masked_fill_nll_loss_forward_11', '''
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
    size_hints=[1, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*i64', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_masked_fill_nll_loss_forward_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clamp_div_masked_fill_nll_loss_forward_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp21 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp4 != tmp3
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.where(tmp5, tmp4, tmp1)
    tmp12 = tl.where(tmp11 < 0, tmp11 + 512, tmp11)
    # tl.device_assert((0 <= tmp12) & (tmp12 < 512), "index out of bounds: 0 <= tmp12 < 512")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (512*r0)), rmask, other=0.0).to(tl.float32)
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp22 = triton_helpers.maximum(tmp21, tmp1)
    tmp23 = triton_helpers.minimum(tmp22, tmp3)
    tmp24 = tmp23 != tmp3
    tmp25 = tmp24.to(tl.int64)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tl.where(tmp24, tmp23, tmp1)
    tmp31 = tl.where(tmp30 < 0, tmp30 + 512, tmp30)
    # tl.device_assert((0 <= tmp31) & (tmp31 < 512), "index out of bounds: 0 <= tmp31 < 512")
    tmp32 = tl.load(in_ptr3 + (tmp31 + (512*r0)), rmask, other=0.0).to(tl.float32)
    tmp33 = -tmp32
    tmp34 = tl.where(tmp24, tmp33, tmp15)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp29.to(tl.float32)
    tmp40 = tmp10.to(tl.float32)
    tmp41 = tmp38 / tmp39
    tmp42 = tmp20 / tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = 2.0
    tmp45 = tmp43 / tmp44
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp39, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp40, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp45, None)
''')
