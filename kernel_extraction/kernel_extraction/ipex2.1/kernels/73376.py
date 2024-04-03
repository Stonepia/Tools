

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/if/cifulsc64b62f2cxjtcegynczyvkb5vhvn4cbzwgykdxx2ccmxeq.py
# Source Nodes: [iadd, l__mod___deberta_embeddings_layer_norm, l__mod___deberta_embeddings_word_embeddings, mul, trampoline_autograd_apply], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.embedding, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.rsub]
# iadd => add
# l__mod___deberta_embeddings_layer_norm => add_1, add_2, convert_element_type, mul, mul_1, rsqrt, sub, var_mean
# l__mod___deberta_embeddings_word_embeddings => embedding
# mul => convert_element_type_1
# trampoline_autograd_apply => convert_element_type_3, convert_element_type_4, full_default_1, lt, mul_3, sub_1, where
triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_1 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*i64', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*fp32', 9: '*i1', 10: '*bf16', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    x2 = xindex % 512
    tmp7_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 128100, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 128100)) | ~xmask, "index out of bounds: 0 <= tmp1 < 128100")
        tmp2 = tl.load(in_ptr1 + (r1 + (1536*tmp1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight,
        )
        tmp7_mean = tl.where(rmask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(rmask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(rmask & xmask, tmp7_weight_next, tmp7_weight)
        tl.store(out_ptr0 + (r1 + (1536*x0)), tmp2, rmask & xmask)
    tmp7_tmp, tmp8_tmp, tmp9_tmp = triton_helpers.welford(
        tmp7_mean, tmp7_m2, tmp7_weight, 1
    )
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = 1536.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-07
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp32 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp38 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + load_seed_offset)
        tmp24 = r1 + (1536*x0)
        tmp25 = tl.rand(tmp23, (tmp24).to(tl.uint32))
        tmp26 = 0.9
        tmp27 = tmp25 < tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = 1.0
        tmp30 = tmp29 - tmp28
        tmp31 = (tmp30 != 0)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 - tmp7
        tmp37 = tmp36 * tmp22
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp37 * tmp39
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp40 + tmp42
        tmp44 = tmp43.to(tl.float32)
        tmp45 = 0.0
        tmp46 = tl.where(tmp31, tmp45, tmp44)
        tmp47 = 1.1111111111111112
        tmp48 = tmp46 * tmp47
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp31, rmask & xmask)
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp48, rmask & xmask)
''')
