

# Original file: ./moondream___60.0/moondream___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ab/cabfu45myxwu6inl4k4baj5k3kbxnkwpozmf7zgeh2ppq7edfrmj.py
# Source Nodes: [add_13, add_14, add_6, add_7, l__self___model_embed_tokens, l__self___model_layers_0_resid_dropout, l__self___model_layers_0_resid_dropout_1, l__self___model_layers_1_resid_dropout, l__self___model_layers_1_resid_dropout_1, l__self___model_layers_2_input_layernorm, l__self___model_layers_2_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.native_layer_norm]
# add_13 => add_17
# add_14 => add_18
# add_6 => add_8
# add_7 => add_9
# l__self___model_embed_tokens => embedding
# l__self___model_layers_0_resid_dropout => clone_3
# l__self___model_layers_0_resid_dropout_1 => clone_4
# l__self___model_layers_1_resid_dropout => clone_7
# l__self___model_layers_1_resid_dropout_1 => clone_8
# l__self___model_layers_2_input_layernorm => add_19, add_20, mul_20, mul_21, rsqrt_2, sub_4, var_mean_2
# l__self___model_layers_2_self_attn_q_proj => convert_element_type_50
triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_11 = async_compile.triton('triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_11', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_embedding_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp14_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp14_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tl.where(tmp8 < 0, tmp8 + 51200, tmp8)
        # tl.device_assert(((0 <= tmp9) & (tmp9 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp9 < 51200")
        tmp10 = tl.load(in_ptr5 + (r1 + (2048*tmp9)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp7 + tmp10
        tmp12 = tmp3 + tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp14_mean_next, tmp14_m2_next, tmp14_weight_next = triton_helpers.welford_reduce(
            tmp13, tmp14_mean, tmp14_m2, tmp14_weight,
        )
        tmp14_mean = tl.where(rmask & xmask, tmp14_mean_next, tmp14_mean)
        tmp14_m2 = tl.where(rmask & xmask, tmp14_m2_next, tmp14_m2)
        tmp14_weight = tl.where(rmask & xmask, tmp14_weight_next, tmp14_weight)
        tl.store(out_ptr0 + (r1 + (2048*x0)), tmp12, rmask & xmask)
    tmp14_tmp, tmp15_tmp, tmp16_tmp = triton_helpers.welford(
        tmp14_mean, tmp14_m2, tmp14_weight, 1
    )
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight,
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp22 - tmp14
        tmp24 = 2048.0
        tmp25 = tmp20 / tmp24
        tmp26 = 1e-05
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (2048*x0)), tmp34, rmask & xmask)
''')
