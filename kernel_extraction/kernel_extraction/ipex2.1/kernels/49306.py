

# Original file: ./DebertaV2ForQuestionAnswering__0_forward_205.0/DebertaV2ForQuestionAnswering__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/po/cpol6yboryql3gwlkrxbr7q2lj7kgqhctq3t2i7gs6nobrtpelvj.py
# Source Nodes: [iadd, l__mod___deberta_embeddings_layer_norm, l__mod___deberta_embeddings_position_embeddings, l__mod___deberta_embeddings_word_embeddings, l__mod___deberta_encoder_layer_0_attention_self_query_proj, mul, trampoline_autograd_apply], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.embedding, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.rsub, aten.view]
# iadd => add
# l__mod___deberta_embeddings_layer_norm => add_1, mul, mul_1, rsqrt, sub, var_mean
# l__mod___deberta_embeddings_position_embeddings => embedding_1
# l__mod___deberta_embeddings_word_embeddings => embedding
# l__mod___deberta_encoder_layer_0_attention_self_query_proj => view
# mul => add_2
# trampoline_autograd_apply => convert_element_type, convert_element_type_1, full_default_1, lt, mul_3, sub_1, where
triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_0 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_0', '''
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_native_layer_norm_backward_rsub_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 128100, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 128100)) | ~xmask, "index out of bounds: 0 <= tmp1 < 128100")
        tmp2 = tl.load(in_ptr1 + (r1 + (1536*tmp1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.where(tmp3 < 0, tmp3 + 512, tmp3)
        # tl.device_assert(((0 <= tmp4) & (tmp4 < 512)) | ~xmask, "index out of bounds: 0 <= tmp4 < 512")
        tmp5 = tl.load(in_ptr3 + (r1 + (1536*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tmp11 = tl.load(in_ptr4 + load_seed_offset)
        tmp12 = r1 + (1536*x0)
        tmp13 = tl.rand(tmp11, (tmp12).to(tl.uint32))
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp13, rmask & xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(out_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = 0.9
        tmp16 = tmp14 < tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = 1.0
        tmp19 = tmp18 - tmp17
        tmp20 = (tmp19 != 0)
        tmp21 = tl.where(tmp0 < 0, tmp0 + 128100, tmp0)
        # tl.device_assert(((0 <= tmp21) & (tmp21 < 128100)) | ~xmask, "index out of bounds: 0 <= tmp21 < 128100")
        tmp22 = tl.load(in_ptr1 + (r1 + (1536*tmp21)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.where(tmp3 < 0, tmp3 + 512, tmp3)
        # tl.device_assert(((0 <= tmp23) & (tmp23 < 512)) | ~xmask, "index out of bounds: 0 <= tmp23 < 512")
        tmp24 = tl.load(in_ptr3 + (r1 + (1536*tmp23)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tmp22 + tmp24
        tmp26 = tmp25 - tmp8
        tmp27 = 1536.0
        tmp28 = tmp9 / tmp27
        tmp29 = 1e-07
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tmp37 = 0.0
        tmp38 = tl.where(tmp20, tmp37, tmp36)
        tmp39 = 1.1111111111111112
        tmp40 = tmp38 * tmp39
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp20, rmask & xmask)
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp32, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp40, rmask & xmask)
    tmp41 = 1536.0
    tmp42 = tmp9 / tmp41
    tmp43 = 1e-07
    tmp44 = tmp42 + tmp43
    tmp45 = libdevice.rsqrt(tmp44)
    tmp46 = tmp45 / tmp41
    tl.store(out_ptr6 + (x0), tmp46, xmask)
''')
