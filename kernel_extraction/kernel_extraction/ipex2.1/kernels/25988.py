

# Original file: ./hf_T5___60.0/hf_T5___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/si/csic3r3kcbfqrscqvfbrlay37zcb2fz5zkndq3pqg4amw52fy7nc.py
# Source Nodes: [add, add_32, add_33, l__mod___model_decoder_block_0_layer_0_dropout, l__mod___model_encoder_embed_tokens, l__mod___model_encoder_embed_tokens_1, mean, mean_14, mul_1, mul_2, mul_35, mul_36, pow_1, pow_15, rsqrt, rsqrt_14], Original ATen: [aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add
# add_32 => add_40
# add_33 => add_41
# l__mod___model_decoder_block_0_layer_0_dropout => clone_35
# l__mod___model_encoder_embed_tokens => embedding
# l__mod___model_encoder_embed_tokens_1 => embedding_2
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
triton_red_fused_add_clone_embedding_mean_mul_pow_rsqrt_4 = async_compile.triton('triton_red_fused_add_clone_embedding_mean_mul_pow_rsqrt_4', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_embedding_mean_mul_pow_rsqrt_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_embedding_mean_mul_pow_rsqrt_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp10 = tl.where(tmp9 < 0, tmp9 + 32128, tmp9)
        # tl.device_assert((0 <= tmp10) & (tmp10 < 32128), "index out of bounds: 0 <= tmp10 < 32128")
        tmp11 = tl.load(in_ptr1 + (r1 + (512*tmp10)), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp17) & (tmp17 < 32128), "index out of bounds: 0 <= tmp17 < 32128")
        tmp18 = tl.load(in_ptr1 + (r1 + (512*tmp17)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = 512.0
        tmp22 = tmp7 / tmp21
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp27 = tmp16 * tmp26
        tmp28 = tl.where(tmp9 < 0, tmp9 + 32128, tmp9)
        # tl.device_assert((0 <= tmp28) & (tmp28 < 32128), "index out of bounds: 0 <= tmp28 < 32128")
        tmp29 = tl.load(in_ptr1 + (r1 + (512*tmp28)), rmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tmp14 / tmp21
        tmp31 = tmp30 + tmp23
        tmp32 = libdevice.rsqrt(tmp31)
        tmp33 = tmp29 * tmp32
        tmp34 = tmp16 * tmp33
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp34, rmask)
''')
