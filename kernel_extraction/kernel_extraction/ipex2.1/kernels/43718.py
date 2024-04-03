

# Original file: ./BlenderbotForCausalLM__52_forward_161.12/BlenderbotForCausalLM__52_forward_161.12.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/3d/c3d74femroropemrzdtl4jef3v4snia7kfyaghi64yp7gl5pkmwy.py
# Source Nodes: [add_1, dropout_1, l__self___fc1, l__self___final_layer_norm], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_1 => add_3
# dropout_1 => gt, mul_3, mul_4
# l__self___fc1 => view_18
# l__self___final_layer_norm => add_4, add_5, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5', '''
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
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (2560*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp6 = tmp4.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = 1.1111111111111112
        tmp10 = tmp8 * tmp9
        tmp11 = tmp5 + tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight,
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
        tl.store(out_ptr1 + (r1 + (2560*x0)), tmp4, rmask & xmask)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tmp25_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(out_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tmp21 = 1.1111111111111112
        tmp22 = tmp20 * tmp21
        tmp23 = tmp16 + tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp25_mean_next, tmp25_m2_next, tmp25_weight_next = triton_helpers.welford_reduce(
            tmp24, tmp25_mean, tmp25_m2, tmp25_weight,
        )
        tmp25_mean = tl.where(rmask & xmask, tmp25_mean_next, tmp25_mean)
        tmp25_m2 = tl.where(rmask & xmask, tmp25_m2_next, tmp25_m2)
        tmp25_weight = tl.where(rmask & xmask, tmp25_weight_next, tmp25_weight)
    tmp25_tmp, tmp26_tmp, tmp27_tmp = triton_helpers.welford(
        tmp25_mean, tmp25_m2, tmp25_weight, 1
    )
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp27 = tmp27_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(out_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp31 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp45 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tmp30 * tmp31
        tmp33 = 1.1111111111111112
        tmp34 = tmp32 * tmp33
        tmp35 = tmp28 + tmp34
        tmp36 = tmp35 - tmp13
        tmp37 = 2560.0
        tmp38 = tmp26 / tmp37
        tmp39 = 1e-05
        tmp40 = tmp38 + tmp39
        tmp41 = libdevice.rsqrt(tmp40)
        tmp42 = tmp36 * tmp41
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tl.store(out_ptr4 + (r1 + (2560*x0)), tmp42, rmask & xmask)
        tl.store(out_ptr5 + (r1 + (2560*x0)), tmp46, rmask & xmask)
    tmp47 = 2560.0
    tmp48 = tmp26 / tmp47
    tmp49 = 1e-05
    tmp50 = tmp48 + tmp49
    tmp51 = libdevice.rsqrt(tmp50)
    tmp52 = tmp51 / tmp47
    tl.store(out_ptr6 + (x0), tmp52, xmask)
''')
