

# Original file: ./timm_vision_transformer_large___60.0/timm_vision_transformer_large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/sv/csvltn2aqwiw6xjk5dbx4gkydhhdwbslgyizk4fjcmg3xp5bjr36.py
# Source Nodes: [add, add_1, add_2, getattr_l__self___blocks___0___attn_proj_drop, getattr_l__self___blocks___0___mlp_drop2, getattr_l__self___blocks___1___attn_qkv, getattr_l__self___blocks___1___norm1], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add => add
# add_1 => add_3
# add_2 => add_7
# getattr_l__self___blocks___0___attn_proj_drop => clone_5
# getattr_l__self___blocks___0___mlp_drop2 => clone_7
# getattr_l__self___blocks___1___attn_qkv => convert_element_type_18
# getattr_l__self___blocks___1___norm1 => add_8, add_9, mul_10, mul_9, rsqrt_2, sub_3, var_mean_2
triton_red_fused__to_copy_add_clone_native_layer_norm_11 = async_compile.triton('triton_red_fused__to_copy_add_clone_native_layer_norm_11', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_native_layer_norm_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8224
    rnumel = 1408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 257
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (1408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 + tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 + tmp7
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
        r2 = rindex
        tmp13 = tl.load(in_ptr0 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r2 + (1408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr3 + (r2 + (1408*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21 - tmp10
        tmp23 = 1408.0
        tmp24 = tmp11 / tmp23
        tmp25 = 1e-06
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp30 = tmp28 * tmp29
        tmp32 = tmp30 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (1408*x3)), tmp33, rmask & xmask)
''')
