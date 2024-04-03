

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/nr/cnre3mobbpenwl3lsnj3pabt5c6o2sywks4uctzqhqnmu5t2aftk.py
# Source Nodes: [iadd, l__self___deberta_embeddings_layer_norm, l__self___deberta_embeddings_word_embeddings, l__self___deberta_encoder_layer_0_attention_self_query_proj, mul, trampoline_autograd_apply], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.embedding, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.rsub, aten.view]
# iadd => add
# l__self___deberta_embeddings_layer_norm => add_1, mul, mul_1, rsqrt, sub, var_mean
# l__self___deberta_embeddings_word_embeddings => embedding
# l__self___deberta_encoder_layer_0_attention_self_query_proj => convert_element_type_2, view
# mul => add_2
# trampoline_autograd_apply => convert_element_type, convert_element_type_1, full_default_1, lt, mul_3, sub_1, where
triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_view_1 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_view_1', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_view_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    x2 = xindex % 512
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 128100, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 128100)) | ~xmask, "index out of bounds: 0 <= tmp1 < 128100")
        tmp2 = tl.load(in_ptr1 + (r1 + (1536*tmp1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(out_ptr0 + (r1 + (1536*x0)), tmp2, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight,
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tmp16 = 1536.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-07
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr2 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + load_seed_offset)
        tmp22 = r1 + (1536*x0)
        tmp23 = tl.rand(tmp21, (tmp22).to(tl.uint32))
        tmp24 = 0.9
        tmp25 = tmp23 < tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 1.0
        tmp28 = tmp27 - tmp26
        tmp29 = (tmp28 != 0)
        tmp32 = tmp30 + tmp31
        tmp33 = tmp32 - tmp6
        tmp34 = tmp33 * tmp20
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = 0.0
        tmp40 = tl.where(tmp29, tmp39, tmp38)
        tmp41 = 1.1111111111111112
        tmp42 = tmp40 * tmp41
        tmp43 = tmp42.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp29, rmask & xmask)
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp42, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (1536*x0)), tmp43, rmask & xmask)
''')
