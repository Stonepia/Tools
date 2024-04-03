

# Original file: ./nanogpt___60.0/nanogpt___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/aw/caw7qmrroah3zb4n3k7suxkmejrjlmh4wcaskbjpwuxshqbmdneh.py
# Source Nodes: [add, l__self___transformer_h_0_attn_c_attn, l__self___transformer_wpe, l__self___transformer_wte, layer_norm], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.native_layer_norm]
# add => add
# l__self___transformer_h_0_attn_c_attn => convert_element_type
# l__self___transformer_wpe => embedding_1
# l__self___transformer_wte => embedding
# layer_norm => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused__to_copy_add_embedding_native_layer_norm_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_native_layer_norm_0', '''
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
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_native_layer_norm_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 50304, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 50304)) | ~xmask, "index out of bounds: 0 <= tmp1 < 50304")
        tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.where(tmp0 < 0, tmp0 + 50304, tmp0)
        # tl.device_assert(((0 <= tmp9) & (tmp9 < 50304)) | ~xmask, "index out of bounds: 0 <= tmp9 < 50304")
        tmp10 = tl.load(in_ptr1 + (r1 + (768*tmp9)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12 - tmp6
        tmp14 = 768.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp24, rmask & xmask)
''')
