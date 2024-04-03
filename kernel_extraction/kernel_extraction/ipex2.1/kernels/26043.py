

# Original file: ./hf_T5___60.0/hf_T5___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/35/c35qsbexdvifc3aiznc7ngwbi2kttkddsxuei2v76r2plwgw5ud7.py
# Source Nodes: [add, add_32, add_33, l__self___model_decoder_block_0_layer_0_dropout, l__self___model_decoder_block_0_layer_1_enc_dec_attention_q, l__self___model_encoder_block_0_layer_0_self_attention_q, l__self___model_encoder_embed_tokens, l__self___model_encoder_embed_tokens_1, mean, mean_14, mul_1, mul_2, mul_35, mul_36, pow_1, pow_15, rsqrt, rsqrt_14], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add
# add_32 => add_40
# add_33 => add_41
# l__self___model_decoder_block_0_layer_0_dropout => clone_35
# l__self___model_decoder_block_0_layer_1_enc_dec_attention_q => convert_element_type_107
# l__self___model_encoder_block_0_layer_0_self_attention_q => convert_element_type
# l__self___model_encoder_embed_tokens => embedding
# l__self___model_encoder_embed_tokens_1 => embedding_2
# mean => mean
# mean_14 => mean_14
# mul_1 => mul_1
# mul_2 => mul_2
# mul_35 => mul_35
# mul_36 => mul_36
# pow_1 => pow_1
# pow_15 => pow_15
# rsqrt => rsqrt
# rsqrt_14 => rsqrt_14
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp16', 3: '*i64', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 + tmp4
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp11 = tl.where(tmp10 < 0, tmp10 + 32128, tmp10)
        # tl.device_assert((0 <= tmp11) & (tmp11 < 32128), "index out of bounds: 0 <= tmp11 < 32128")
        tmp12 = tl.load(in_ptr1 + (r1 + (512*tmp11)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp18) & (tmp18 < 32128), "index out of bounds: 0 <= tmp18 < 32128")
        tmp19 = tl.load(in_ptr1 + (r1 + (512*tmp18)), rmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp19 + tmp21
        tmp23 = 512.0
        tmp24 = tmp8 / tmp23
        tmp25 = 1e-06
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp29 = tmp17 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tl.where(tmp10 < 0, tmp10 + 32128, tmp10)
        # tl.device_assert((0 <= tmp31) & (tmp31 < 32128), "index out of bounds: 0 <= tmp31 < 32128")
        tmp32 = tl.load(in_ptr1 + (r1 + (512*tmp31)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tmp15 / tmp23
        tmp34 = tmp33 + tmp25
        tmp35 = libdevice.rsqrt(tmp34)
        tmp36 = tmp32 * tmp35
        tmp37 = tmp17 * tmp36
        tmp38 = tmp37.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp30, rmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp38, rmask)
''')