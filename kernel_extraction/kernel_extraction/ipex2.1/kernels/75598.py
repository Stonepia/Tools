

# Original file: ./AlbertForQuestionAnswering__0_forward_205.0/AlbertForQuestionAnswering__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/2v/c2vg2dzqed4tvffadgmqqhluheq5c35gzjqnzupncsnevunr3z35.py
# Source Nodes: [add_2, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_5
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout => clone_6
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_ffn => view_20
triton_red_fused_add_clone_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_native_layer_norm_backward_view_6', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_native_layer_norm_backward_view_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_native_layer_norm_backward_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tmp4_mean_next
        tmp4_m2 = tmp4_m2_next
        tmp4_weight = tmp4_weight_next
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
        tmp7 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp8 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp17 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 4096.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-12
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp16, None)
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp20, None)
    tmp21 = 4096.0
    tmp22 = tmp5 / tmp21
    tmp23 = 1e-12
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp25 / tmp21
    tl.store(out_ptr4 + (x0), tmp26, None)
''')
