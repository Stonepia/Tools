

# Original file: ./DebertaForMaskedLM__0_forward_133.0/DebertaForMaskedLM__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/zj/czjf6h6rwpzp2xukrplhkcnclsjvkaryqgr3khcs52acona2h6ng.py
# Source Nodes: [add_4, add_5, add_6, l__self___deberta_encoder_layer_0_intermediate_dense, mean_2, mean_3, mul_4, pow_2, sqrt_2, sub_2, trampoline_autograd_apply_1, trampoline_autograd_apply_3, truediv_2], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
# add_4 => add_5
# add_5 => add_6
# add_6 => add_7
# l__self___deberta_encoder_layer_0_intermediate_dense => convert_element_type_16, view_14
# mean_2 => mean_2
# mean_3 => mean_3
# mul_4 => mul_7
# pow_2 => pow_2
# sqrt_2 => sqrt_2
# sub_2 => sub_6
# trampoline_autograd_apply_1 => full_default_5
# trampoline_autograd_apply_3 => convert_element_type_14, convert_element_type_15, lt_2, mul_6, sub_5, where_4
# truediv_2 => div_3
triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_10 = async_compile.triton('triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_10', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp9 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.9
    tmp4 = tmp2 < tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 1.0
    tmp7 = tmp6 - tmp5
    tmp8 = (tmp7 != 0)
    tmp10 = 0.0
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 768.0
    tmp22 = tmp20 / tmp21
    tmp23 = tmp16 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp28 / tmp21
    tmp30 = 1e-07
    tmp31 = tmp29 + tmp30
    tmp32 = tl.sqrt(tmp31)
    tmp34 = tmp23 / tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp23, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp38, rmask)
''')
