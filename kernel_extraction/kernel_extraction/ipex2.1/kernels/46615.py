

# Original file: ./moondream___60.0/moondream___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/np/cnpb5jpfczapl5zt66ub5k6tazxoe7aofz2cviywyreoutvhbuun.py
# Source Nodes: [add_6, add_7, l__self___model_embed_tokens, l__self___model_layers_0_resid_dropout, l__self___model_layers_0_resid_dropout_1, l__self___model_layers_1_input_layernorm, l__self___model_layers_1_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.native_layer_norm]
# add_6 => add_8
# add_7 => add_9
# l__self___model_embed_tokens => embedding
# l__self___model_layers_0_resid_dropout => clone_3
# l__self___model_layers_0_resid_dropout_1 => clone_4
# l__self___model_layers_1_input_layernorm => add_10, add_11, mul_10, mul_11, rsqrt_1, sub_2, var_mean_1
# l__self___model_layers_1_self_attn_q_proj => convert_element_type_25
triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_10 = async_compile.triton('triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_10', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tl.where(tmp4 < 0, tmp4 + 51200, tmp4)
        # tl.device_assert(((0 <= tmp5) & (tmp5 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp5 < 51200")
        tmp6 = tl.load(in_ptr3 + (r1 + (2048*tmp5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp3 + tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight,
        )
        tmp9_mean = tl.where(rmask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(rmask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(rmask & xmask, tmp9_weight_next, tmp9_weight)
    tmp9_tmp, tmp10_tmp, tmp11_tmp = triton_helpers.welford(
        tmp9_mean, tmp9_m2, tmp9_weight, 1
    )
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.where(tmp4 < 0, tmp4 + 51200, tmp4)
        # tl.device_assert(((0 <= tmp16) & (tmp16 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp16 < 51200")
        tmp17 = tl.load(in_ptr3 + (r1 + (2048*tmp16)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp15 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = 2048.0
        tmp21 = tmp10 / tmp20
        tmp22 = 1e-05
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tmp29.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp30, rmask & xmask)
''')
