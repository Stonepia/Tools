

# Original file: ./DebertaForMaskedLM__0_forward_133.0/DebertaForMaskedLM__0_forward_133.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/j7/cj7lxinwp4wlrzs5ksnyut3cz3xcecj5dmw7lexyvm5uddp4fqbk.py
# Source Nodes: [add, iadd, l__mod___deberta_embeddings_word_embeddings, mean, mean_1, mul, mul_1, pow_1, sqrt, sub, trampoline_autograd_apply, truediv], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.div, aten.embedding, aten.masked_fill, aten.mean, aten.mul, aten.pow, aten.rsub, aten.sqrt, aten.sub]
# add => add_1
# iadd => add
# l__mod___deberta_embeddings_word_embeddings => embedding
# mean => mean
# mean_1 => mean_1
# mul => mul
# mul_1 => add_2
# pow_1 => pow_1
# sqrt => sqrt
# sub => sub
# trampoline_autograd_apply => convert_element_type, convert_element_type_1, full_default_1, lt, mul_2, sub_2, where
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bernoulli_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
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
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 50265, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 50265), "index out of bounds: 0 <= tmp1 < 50265")
    tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = 768.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp4 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp16 / tmp9
    tmp18 = 1e-07
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = tl.load(in_ptr3 + load_seed_offset)
    tmp22 = r1 + (768*x0)
    tmp23 = tl.rand(tmp21, (tmp22).to(tl.uint32))
    tmp24 = 0.9
    tmp25 = tmp23 < tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = 1.0
    tmp28 = tmp27 - tmp26
    tmp29 = (tmp28 != 0)
    tmp31 = tmp11 / tmp20
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = 0.0
    tmp36 = tl.where(tmp29, tmp35, tmp34)
    tmp37 = 1.1111111111111112
    tmp38 = tmp36 * tmp37
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp20, None)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp38, rmask)
''')
