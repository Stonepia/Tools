

# Original file: ./moondream___60.0/moondream___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/br/cbraoof3wzyusvs2cokwbmz3mnt2cugwlag3w3qfrwdj5concdql.py
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
# l__mod___model_layers_2_input_layernorm => add_19, add_20, convert_element_type_11, convert_element_type_12, mul_20, mul_21, rsqrt_2, sub_4, var_mean_2
triton_red_fused_add_clone_embedding_native_layer_norm_11 = async_compile.triton('triton_red_fused_add_clone_embedding_native_layer_norm_11', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*i64', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_embedding_native_layer_norm_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_embedding_native_layer_norm_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tl.where(tmp6 < 0, tmp6 + 51200, tmp6)
        # tl.device_assert(((0 <= tmp7) & (tmp7 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp7 < 51200")
        tmp8 = tl.load(in_ptr4 + (r1 + (2048*tmp7)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp5 + tmp8
        tmp10 = tmp2 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight,
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp10, rmask & xmask)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
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
        tmp22 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp13
        tmp25 = 2048.0
        tmp26 = tmp20 / tmp25
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp37, rmask & xmask)
''')
