

# Original file: ./hf_T5_base___60.0/hf_T5_base___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ha/chafi5jzu25djnltubsyxaybvpqynldxtqzabqoj7hmauhpshmua.py
# Source Nodes: [add, add_56, add_57, l__mod___model_decoder_block_0_layer_0_dropout, l__mod___model_encoder_embed_tokens, l__mod___model_encoder_embed_tokens_1, mean, mean_26, mul_1, mul_2, mul_59, mul_60, pow_1, pow_27, rsqrt, rsqrt_26, to_1, to_2, to_59, to_60], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add
# add_56 => add_70
# add_57 => add_71
# l__mod___model_decoder_block_0_layer_0_dropout => clone_65
# l__mod___model_encoder_embed_tokens => embedding
# l__mod___model_encoder_embed_tokens_1 => embedding_2
# mean => mean
# mean_26 => mean_26
# mul_1 => mul_1
# mul_2 => mul_2
# mul_59 => mul_59
# mul_60 => mul_60
# pow_1 => pow_1
# pow_27 => pow_27
# rsqrt => rsqrt
# rsqrt_26 => rsqrt_26
# to_1 => convert_element_type_1
# to_2 => convert_element_type_2
# to_59 => convert_element_type_87
# to_60 => convert_element_type_88
triton_red_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_4 = async_compile.triton('triton_red_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_4', '''
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*i64', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp11 = tl.where(tmp10 < 0, tmp10 + 32128, tmp10)
        # tl.device_assert((0 <= tmp11) & (tmp11 < 32128), "index out of bounds: 0 <= tmp11 < 32128")
        tmp12 = tl.load(in_ptr1 + (r1 + (768*tmp11)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp19) & (tmp19 < 32128), "index out of bounds: 0 <= tmp19 < 32128")
        tmp20 = tl.load(in_ptr1 + (r1 + (768*tmp19)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 768.0
        tmp25 = tmp8 / tmp24
        tmp26 = 1e-06
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp18 * tmp30
        tmp32 = tl.where(tmp10 < 0, tmp10 + 32128, tmp10)
        # tl.device_assert((0 <= tmp32) & (tmp32 < 32128), "index out of bounds: 0 <= tmp32 < 32128")
        tmp33 = tl.load(in_ptr1 + (r1 + (768*tmp32)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp16 / tmp24
        tmp36 = tmp35 + tmp26
        tmp37 = libdevice.rsqrt(tmp36)
        tmp38 = tmp34 * tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp18 * tmp39
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
        tl.store(out_ptr3 + (r1 + (768*x0)), tmp40, rmask)
''')
