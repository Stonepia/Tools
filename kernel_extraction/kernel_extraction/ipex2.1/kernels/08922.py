

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/ja/cjajsfs5hb2e3teviy2nviy57dwgx4aeoi53s2uof4c76fiqdfge.py
# Source Nodes: [add_5, add_6, add_7, getattr_getattr_l__self___levels___1___transformer_encoder___0___attn_proj_drop, getattr_getattr_l__self___levels___1___transformer_encoder___0___mlp_drop2, getattr_getattr_l__self___levels___1___transformer_encoder___1___attn_qkv, layer_norm_7], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_5 => add_17
# add_6 => add_20
# add_7 => add_24
# getattr_getattr_l__self___levels___1___transformer_encoder___0___attn_proj_drop => clone_21
# getattr_getattr_l__self___levels___1___transformer_encoder___0___mlp_drop2 => clone_23
# getattr_getattr_l__self___levels___1___transformer_encoder___1___attn_qkv => convert_element_type_50
# layer_norm_7 => add_25, add_26, mul_29, mul_30, rsqrt_7, sub_10, var_mean_7
triton_red_fused__to_copy_add_clone_native_layer_norm_23 = async_compile.triton('triton_red_fused__to_copy_add_clone_native_layer_norm_23', '''
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
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_native_layer_norm_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196) % 4
    x2 = (xindex // 784)
    x4 = xindex % 784
    x5 = xindex
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x1 % 2)) + (28*(x0 // 14)) + (392*(x1 // 2)) + (784*r3) + (200704*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3 + (256*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r3 + (256*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 + tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight,
        )
        tmp11_mean = tl.where(rmask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask & xmask, tmp11_weight_next, tmp11_weight)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp14 = tl.load(in_ptr0 + ((14*(x1 % 2)) + (28*(x0 // 14)) + (392*(x1 // 2)) + (784*r3) + (200704*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r3 + (256*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (r3 + (256*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23 - tmp11
        tmp25 = 256.0
        tmp26 = tmp12 / tmp25
        tmp27 = 1e-06
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr2 + (r3 + (256*x5)), tmp35, rmask & xmask)
''')
