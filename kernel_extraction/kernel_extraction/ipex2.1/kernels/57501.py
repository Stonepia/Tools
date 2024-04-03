

# Original file: ./DebertaForMaskedLM__0_backward_135.1/DebertaForMaskedLM__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/35/c35mlhd5p4mhlu5g4e63g4npgmu2zx6qfbv56dr3cln2rq6vodek.py
# Source Nodes: [iadd, sub, trampoline_autograd_apply, truediv], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sub, aten.sum]
# iadd => add
# sub => sub
# trampoline_autograd_apply => full_default_1
# truediv => div
triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_32 = async_compile.triton('triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_32', '''
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
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr5'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask)
    tmp1 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr5 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp0, tmp5, tmp4)
    tmp7 = 1.1111111111111112
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = -tmp10
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 - tmp15
    tmp18 = tmp16 / tmp17
    tmp19 = tmp18 / tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp10 / tmp17
    tmp26 = 2.0
    tmp27 = tmp17 * tmp26
    tmp28 = tmp24 / tmp27
    tmp29 = 768.0
    tmp30 = tmp28 / tmp29
    tmp31 = tmp16 * tmp26
    tmp32 = tmp30 * tmp31
    tmp33 = -tmp25
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = -tmp32
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp44 = tl.where(tmp43 < 0, tmp43 + 50265, tmp43)
    tmp45 = tl.full([1], 0, tl.int64)
    tmp46 = tmp43 == tmp45
    tmp47 = tmp25 + tmp32
    tmp48 = tmp37 + tmp42
    tmp49 = tmp48 / tmp29
    tmp50 = tmp47 + tmp49
    tmp51 = tl.where(tmp46, tmp5, tmp50)
    tl.store(out_ptr1 + (r2 + (768*x3)), tmp25, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp32, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r2 + (768*tmp44), [RBLOCK])), tmp51, rmask)
    tl.store(out_ptr3 + (x3), tmp37, None)
    tl.store(out_ptr4 + (x3), tmp42, None)
''')
