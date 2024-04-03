

# Original file: ./AlbertForMaskedLM__0_forward_205.0/AlbertForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/3y/c3ycnxov5y7l6mhftlrzywyhabtgyevwq7wodd4tngl7umqiiu5b.py
# Source Nodes: [add_5, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query_1, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_5 => add_10
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm => add_7, mul_4
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query_1 => view_24
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm => add_11, add_12, mul_10, mul_9, rsqrt_2, sub_4, var_mean_2
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_8', '''
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
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
        tmp3 = tmp1 * tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = tmp0 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tmp8_mean_next
        tmp8_m2 = tmp8_m2_next
        tmp8_weight = tmp8_weight_next
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp12 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp13 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
        tmp25 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
        tmp27 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 + tmp16
        tmp18 = tmp17 - tmp8
        tmp19 = 4096.0
        tmp20 = tmp9 / tmp19
        tmp21 = 1e-12
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp24, None)
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp28, None)
    tmp29 = 4096.0
    tmp30 = tmp9 / tmp29
    tmp31 = 1e-12
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp33 / tmp29
    tl.store(out_ptr4 + (x0), tmp34, None)
''')
