

# Original file: ./DebertaForQuestionAnswering__0_forward_133.0/DebertaForQuestionAnswering__0_forward_133.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/zq/czqkthnkfqag2fz7lcqlqdnx7lke36mb5tmrxqlwxp4c4wplokf2.py
# Source Nodes: [add_4, add_5, add_6, l__mod___deberta_encoder_layer_0_intermediate_dense, mean_2, mean_3, mul_4, pow_2, sqrt_2, sub_2, trampoline_autograd_apply, trampoline_autograd_apply_3, truediv_2], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.div, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub, aten.view]
# add_4 => add_5
# add_5 => add_6
# add_6 => add_7
# l__mod___deberta_encoder_layer_0_intermediate_dense => view_14
# mean_2 => mean_2
# mean_3 => mean_3
# mul_4 => mul_7
# pow_2 => pow_2
# sqrt_2 => sqrt_2
# sub_2 => sub_6
# trampoline_autograd_apply => full_default_1
# trampoline_autograd_apply_3 => convert_element_type_5, convert_element_type_6, lt_2, mul_6, sub_5, where_4
# truediv_2 => div_3
triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_7 = async_compile.triton('triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_7', '''
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
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_bernoulli_div_masked_fill_mean_mul_pow_rsub_sqrt_sub_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
    xnumel = 8192
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
    tmp9 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp32 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 768.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp15 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp27 / tmp20
    tmp29 = 1e-07
    tmp30 = tmp28 + tmp29
    tmp31 = tl.sqrt(tmp30)
    tmp33 = tmp22 / tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, None)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp36, rmask)
''')
