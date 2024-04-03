

# Original file: ./DebertaV2ForMaskedLM__0_forward_205.0/DebertaV2ForMaskedLM__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/5t/c5tinghfza6mytaqjt2qm5okzfbftqjj46ktchicsxapx7tv6wyi.py
# Source Nodes: [gelu_24, l__mod___cls_predictions_decoder, l__mod___cls_predictions_transform_layer_norm], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# gelu_24 => add_171, erf_24, mul_269, mul_270, mul_271
# l__mod___cls_predictions_decoder => view_530
# l__mod___cls_predictions_transform_layer_norm => add_172, add_173, mul_272, mul_273, rsqrt_49, sub_146, var_mean_49
triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.triton('triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_9', '''
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_9(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.5
        tmp2 = tmp0 * tmp1
        tmp3 = 0.7071067811865476
        tmp4 = tmp0 * tmp3
        tmp5 = libdevice.erf(tmp4)
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = 0.5
        tmp15 = tmp13 * tmp14
        tmp16 = 0.7071067811865476
        tmp17 = tmp13 * tmp16
        tmp18 = libdevice.erf(tmp17)
        tmp19 = 1.0
        tmp20 = tmp18 + tmp19
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21 - tmp10
        tmp23 = 1536.0
        tmp24 = tmp11 / tmp23
        tmp25 = 1e-07
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp30 = tmp28 * tmp29
        tmp32 = tmp30 + tmp31
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp28, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp32, rmask & xmask)
    tmp33 = 1536.0
    tmp34 = tmp11 / tmp33
    tmp35 = 1e-07
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp37 / tmp33
    tl.store(out_ptr4 + (x0), tmp38, xmask)
''')
