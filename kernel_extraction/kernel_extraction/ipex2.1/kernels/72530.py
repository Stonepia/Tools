

# Original file: ./timm_vision_transformer_large___60.0/timm_vision_transformer_large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/72/c72pjyezhrt4avv7t7ev2mzggj3rkjk5ic3jdzjhxhh4k5eni7d3.py
# Source Nodes: [add, add_1, add_2, add_3, getattr_l__self___blocks___0___attn_proj_drop, getattr_l__self___blocks___0___mlp_drop2, getattr_l__self___blocks___1___attn_proj_drop, getattr_l__self___blocks___1___mlp_fc1, getattr_l__self___blocks___1___norm2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add => add
# add_1 => add_3
# add_2 => add_7
# add_3 => add_10
# getattr_l__self___blocks___0___attn_proj_drop => clone_5
# getattr_l__self___blocks___0___mlp_drop2 => clone_7
# getattr_l__self___blocks___1___attn_proj_drop => clone_12
# getattr_l__self___blocks___1___mlp_fc1 => convert_element_type_25
# getattr_l__self___blocks___1___norm2 => add_11, add_12, mul_13, mul_14, rsqrt_3, sub_5, var_mean_3
triton_red_fused__to_copy_add_clone_native_layer_norm_12 = async_compile.triton('triton_red_fused__to_copy_add_clone_native_layer_norm_12', '''
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
    size_hints=[16384, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_native_layer_norm_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8224
    rnumel = 1408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 257
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (1408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr4 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 + tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 + tmp7
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 + tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight,
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
        tl.store(out_ptr0 + (r2 + (1408*x3)), tmp11, rmask & xmask)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp16 = tl.load(out_ptr0 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight,
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(out_ptr0 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp21 - tmp13
        tmp23 = 1408.0
        tmp24 = tmp19 / tmp23
        tmp25 = 1e-06
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp30 = tmp28 * tmp29
        tmp32 = tmp30 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (1408*x3)), tmp33, rmask & xmask)
''')
