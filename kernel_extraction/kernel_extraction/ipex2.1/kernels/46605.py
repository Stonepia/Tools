

# Original file: ./moondream___60.0/moondream___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/o3/co3easee7wnty27l57mwx3v2gm7mg2hankchyewwq4wkt22vo26v.py
# Source Nodes: [l__self___model_embed_tokens, l__self___model_layers_0_input_layernorm, l__self___model_layers_0_self_attn_q_proj], Original ATen: [aten._to_copy, aten.embedding, aten.native_layer_norm]
# l__self___model_embed_tokens => embedding
# l__self___model_layers_0_input_layernorm => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# l__self___model_layers_0_self_attn_q_proj => convert_element_type
triton_red_fused__to_copy_embedding_native_layer_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_native_layer_norm_0', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_native_layer_norm_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 51200, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp1 < 51200")
        tmp2 = tl.load(in_ptr1 + (r1 + (2048*tmp1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp0 < 0, tmp0 + 51200, tmp0)
        # tl.device_assert(((0 <= tmp7) & (tmp7 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp7 < 51200")
        tmp8 = tl.load(in_ptr1 + (r1 + (2048*tmp7)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp8 - tmp4
        tmp10 = 2048.0
        tmp11 = tmp5 / tmp10
        tmp12 = 1e-05
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp9 * tmp14
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp20, rmask & xmask)
''')
