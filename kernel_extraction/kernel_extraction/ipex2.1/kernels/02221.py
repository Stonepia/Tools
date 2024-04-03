

# Original file: ./AlbertForQuestionAnswering__0_forward_133.0/AlbertForQuestionAnswering__0_forward_133.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/47/c47kmlijgzdt74uza7ldpothkkfk7e4h3ausnj4vr4x5ysx34yt3.py
# Source Nodes: [add_2, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_2 => add_5
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm => add_6, add_7, convert_element_type_5, convert_element_type_6, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_output_dropout => clone_6
triton_red_fused_add_clone_native_layer_norm_5 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_5', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_5(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp1 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tmp5_mean_next
        tmp5_m2 = tmp5_m2_next
        tmp5_weight = tmp5_weight_next
        tl.store(in_out_ptr0 + (r1 + (4096*x0)), tmp2, None)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp5, None)
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight,
        )
        tmp11_mean = tmp11_mean_next
        tmp11_m2 = tmp11_m2_next
        tmp11_weight = tmp11_weight_next
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = 4096.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-12
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_out_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp5
        tmp22 = tmp21 * tmp18
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (4096*x0)), tmp29, None)
''')
