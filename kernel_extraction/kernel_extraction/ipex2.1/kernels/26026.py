

# Original file: ./hf_T5___60.0/hf_T5___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/t7/ct76uhut7lpf23c3bmrgq4fqqucktizrpurti2vnt4y6yuetitse.py
# Source Nodes: [add, add_32, add_33, clamp_12, l__mod___model_decoder_block_0_layer_0_dropout, l__mod___model_encoder_embed_tokens, l__mod___model_encoder_embed_tokens_1, mean, mean_14, mul_1, mul_2, mul_35, mul_36, neg_13, pow_1, pow_15, rsqrt, rsqrt_14, to_1, to_2, to_35, to_36, where_1, where_14], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.clone, aten.embedding, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add => add
# add_32 => add_40
# add_33 => add_41
# clamp_12 => clamp_max_12, clamp_min_12, convert_element_type_75, convert_element_type_76
# l__mod___model_decoder_block_0_layer_0_dropout => clone_35
# l__mod___model_encoder_embed_tokens => embedding
# l__mod___model_encoder_embed_tokens_1 => embedding_2
# mean => mean
# mean_14 => mean_14
# mul_1 => mul_1
# mul_2 => mul_2
# mul_35 => mul_35
# mul_36 => mul_36
# neg_13 => neg_13
# pow_1 => pow_1
# pow_15 => pow_15
# rsqrt => rsqrt
# rsqrt_14 => rsqrt_14
# to_1 => convert_element_type_1
# to_2 => convert_element_type_2
# to_35 => convert_element_type_77
# to_36 => convert_element_type_78
# where_1 => full_default_2, full_default_3
# where_14 => where_14
triton_red_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_6 = async_compile.triton('triton_red_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_6', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*i1', 4: '*i64', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clamp_clone_embedding_mean_mul_neg_pow_rsqrt_scalar_tensor_where_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 64504.0
        tmp9 = 65504.0
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp11 = -tmp10
        tmp12 = triton_helpers.maximum(tmp5, tmp11)
        tmp13 = triton_helpers.minimum(tmp12, tmp10)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp21 = tl.where(tmp20 < 0, tmp20 + 32128, tmp20)
        # tl.device_assert((0 <= tmp21) & (tmp21 < 32128), "index out of bounds: 0 <= tmp21 < 32128")
        tmp22 = tl.load(in_ptr1 + (r1 + (512*tmp21)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp34 = tl.load(in_ptr3 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp29) & (tmp29 < 32128), "index out of bounds: 0 <= tmp29 < 32128")
        tmp30 = tl.load(in_ptr1 + (r1 + (512*tmp29)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tmp30 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp36 = 64504.0
        tmp37 = 65504.0
        tmp38 = tl.where(tmp35, tmp36, tmp37)
        tmp39 = -tmp38
        tmp40 = triton_helpers.maximum(tmp33, tmp39)
        tmp41 = triton_helpers.minimum(tmp40, tmp38)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = 512.0
        tmp45 = tmp18 / tmp44
        tmp46 = 1e-06
        tmp47 = tmp45 + tmp46
        tmp48 = libdevice.rsqrt(tmp47)
        tmp49 = tmp43 * tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp28 * tmp50
        tmp52 = tl.where(tmp20 < 0, tmp20 + 32128, tmp20)
        # tl.device_assert((0 <= tmp52) & (tmp52 < 32128), "index out of bounds: 0 <= tmp52 < 32128")
        tmp53 = tl.load(in_ptr1 + (r1 + (512*tmp52)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp26 / tmp44
        tmp56 = tmp55 + tmp46
        tmp57 = libdevice.rsqrt(tmp56)
        tmp58 = tmp54 * tmp57
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp28 * tmp59
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp51, rmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp60, rmask)
''')
