

# Original file: ./DebertaV2ForQuestionAnswering__0_forward_205.0/DebertaV2ForQuestionAnswering__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/rn/crnaubix6xelflukev226jp3hxv76d4fimubzh3rvm2fbwgpsrv6.py
# Source Nodes: [iadd, l__mod___deberta_embeddings_layer_norm, l__mod___deberta_embeddings_position_embeddings, l__mod___deberta_embeddings_word_embeddings, mul, trampoline_autograd_apply], Original ATen: [aten._to_copy, aten.add, aten.bernoulli, aten.embedding, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.rsub]
# iadd => add
# l__mod___deberta_embeddings_layer_norm => add_1, add_2, convert_element_type, mul, mul_1, rsqrt, sub, var_mean
# l__mod___deberta_embeddings_position_embeddings => embedding_1
# l__mod___deberta_embeddings_word_embeddings => embedding
# mul => convert_element_type_1
# trampoline_autograd_apply => convert_element_type_3, convert_element_type_4, full_default_1, lt, mul_3, sub_1, where
triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_0 = async_compile.triton('triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_0', '''
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
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp16', 3: '*i64', 4: '*fp16', 5: '*i64', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp32', 10: '*i1', 11: '*fp16', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_bernoulli_embedding_masked_fill_mul_native_layer_norm_rsub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 128100, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 128100)) | ~xmask, "index out of bounds: 0 <= tmp1 < 128100")
        tmp2 = tl.load(in_ptr1 + (r1 + (1536*tmp1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.where(tmp3 < 0, tmp3 + 512, tmp3)
        # tl.device_assert(((0 <= tmp4) & (tmp4 < 512)) | ~xmask, "index out of bounds: 0 <= tmp4 < 512")
        tmp5 = tl.load(in_ptr3 + (r1 + (1536*tmp4)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight,
        )
        tmp9_mean = tl.where(rmask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(rmask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(rmask & xmask, tmp9_weight_next, tmp9_weight)
        tl.store(out_ptr0 + (r1 + (1536*x0)), tmp6, rmask & xmask)
    tmp9_tmp, tmp10_tmp, tmp11_tmp = triton_helpers.welford(
        tmp9_mean, tmp9_m2, tmp9_weight, 1
    )
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr4 + load_seed_offset)
        tmp24 = r1 + (1536*x0)
        tmp25 = tl.rand(tmp23, (tmp24).to(tl.uint32))
        tmp26 = 0.9
        tmp27 = tmp25 < tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = 1.0
        tmp30 = tmp29 - tmp28
        tmp31 = (tmp30 != 0)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33 - tmp9
        tmp35 = tmp34 * tmp22
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 * tmp37
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp38 + tmp40
        tmp42 = tmp41.to(tl.float32)
        tmp43 = 0.0
        tmp44 = tl.where(tmp31, tmp43, tmp42)
        tmp45 = 1.1111111111111112
        tmp46 = tmp44 * tmp45
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp31, rmask & xmask)
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp46, rmask & xmask)
''')
