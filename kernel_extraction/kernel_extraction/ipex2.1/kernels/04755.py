

# Original file: ./DebertaForMaskedLM__0_forward_133.0/DebertaForMaskedLM__0_forward_133.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/3s/c3sbc4jumtguhyeopqng4tautw4vp6wwjlwc2xq7jvjsrxgy4b5u.py
# Source Nodes: [add, float_1, iadd, l__mod___deberta_embeddings_word_embeddings, mean, mean_1, mul, mul_1, pow_1, sqrt, sub, to, trampoline_autograd_apply, truediv], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.div, aten.embedding, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub]
# add => add_1
# float_1 => convert_element_type
# iadd => add
# l__mod___deberta_embeddings_word_embeddings => embedding
# mean => mean
# mean_1 => mean_1
# mul => mul
# mul_1 => add_2
# pow_1 => pow_1
# sqrt => sqrt
# sub => sub
# to => convert_element_type_1
# trampoline_autograd_apply => convert_element_type_3, convert_element_type_4, full_default_1, lt, mul_2, sub_2, where
# truediv => div
triton_per_fused__to_copy_add_bernoulli_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1 = async_compile.triton('triton_per_fused__to_copy_add_bernoulli_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*i64', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*i1', 10: '*bf16', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bernoulli_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_bernoulli_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
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
    x0 = xindex
    r1 = rindex
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 50265, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 50265), "index out of bounds: 0 <= tmp1 < 50265")
    tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = 768.0
    tmp11 = tmp9 / tmp10
    tmp12 = tmp5 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp17 / tmp10
    tmp19 = 1e-07
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sqrt(tmp20)
    tmp22 = tl.load(in_ptr3 + load_seed_offset)
    tmp23 = r1 + (768*x0)
    tmp24 = tl.rand(tmp22, (tmp23).to(tl.uint32))
    tmp25 = 0.9
    tmp26 = tmp24 < tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.0
    tmp29 = tmp28 - tmp27
    tmp30 = (tmp29 != 0)
    tmp32 = tmp12 / tmp21
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = 0.0
    tmp38 = tl.where(tmp30, tmp37, tmp36)
    tmp39 = 1.1111111111111112
    tmp40 = tmp38 * tmp39
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp21, None)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp30, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp40, rmask)
''')
