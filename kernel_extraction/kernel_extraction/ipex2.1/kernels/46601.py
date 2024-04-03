

# Original file: ./moondream___60.0/moondream___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/w3/cw3vm6r5dos3dxkzwpea3t3qrbzxw6m5mcrj25bwriqxja6j6uv5.py
# Source Nodes: [add_13, add_14, add_6, add_7, l__mod___model_embed_tokens, l__mod___model_layers_0_resid_dropout, l__mod___model_layers_0_resid_dropout_1, l__mod___model_layers_1_resid_dropout, l__mod___model_layers_1_resid_dropout_1, l__mod___model_layers_2_input_layernorm], Original ATen: [aten.add, aten.clone, aten.embedding, aten.native_layer_norm]
# add_13 => add_17
# add_14 => add_18
# add_6 => add_8
# add_7 => add_9
# l__mod___model_embed_tokens => embedding
# l__mod___model_layers_0_resid_dropout => clone_3
# l__mod___model_layers_0_resid_dropout_1 => clone_4
# l__mod___model_layers_1_resid_dropout => clone_7
# l__mod___model_layers_1_resid_dropout_1 => clone_8
# l__mod___model_layers_2_input_layernorm => add_19, add_20, mul_20, mul_21, rsqrt_2, sub_4, var_mean_2
triton_red_fused_add_clone_embedding_native_layer_norm_10 = async_compile.triton('triton_red_fused_add_clone_embedding_native_layer_norm_10', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_embedding_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_embedding_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tl.where(tmp6 < 0, tmp6 + 51200, tmp6)
        # tl.device_assert(((0 <= tmp7) & (tmp7 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp7 < 51200")
        tmp8 = tl.load(in_ptr4 + (r1 + (2048*tmp7)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp5 + tmp8
        tmp10 = tmp2 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight,
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp10, rmask & xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight,
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp20 - tmp12
        tmp22 = 2048.0
        tmp23 = tmp18 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp31, rmask & xmask)
''')
