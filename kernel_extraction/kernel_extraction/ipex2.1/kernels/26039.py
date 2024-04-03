

# Original file: ./hf_T5___60.0/hf_T5___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/7e/c7eusetzac5jcgnriwjg7fnubmp22p72mzvfkwx4elhwsb5ueuwi.py
# Source Nodes: [add_28, l__self___model_decoder_block_0_layer_0_self_attention_q, l__self___model_encoder_embed_tokens_1, mean_13, mul_32, mul_33, pow_14, rsqrt_13], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_28 => add_35
# l__self___model_decoder_block_0_layer_0_self_attention_q => convert_element_type_95
# l__self___model_encoder_embed_tokens_1 => embedding_2
# mean_13 => mean_13
# mul_32 => mul_32
# mul_33 => mul_33
# pow_14 => pow_14
# rsqrt_13 => rsqrt_13
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp8) & (tmp8 < 32128), "index out of bounds: 0 <= tmp8 < 32128")
        tmp9 = tl.load(in_ptr1 + (r1 + (512*tmp8)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = 512.0
        tmp11 = tmp5 / tmp10
        tmp12 = 1e-06
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp9 * tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp17, rmask)
''')
