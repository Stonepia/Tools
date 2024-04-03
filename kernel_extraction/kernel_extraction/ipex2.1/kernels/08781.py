

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/i2/ci2agld7xar7uvvnzvwcdjnvqx7sd4ouqbwle5e66htwh3prxvjc.py
# Source Nodes: [add_5, getattr_getattr_l__self___levels___1___transformer_encoder___0___attn_qkv, layer_norm_5], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
# add_5 => add_17
# getattr_getattr_l__self___levels___1___transformer_encoder___0___attn_qkv => convert_element_type_36
# layer_norm_5 => add_18, add_19, mul_20, mul_21, rsqrt_5, sub_7, var_mean_5
triton_red_fused__to_copy_add_native_layer_norm_15 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_15', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x1 % 2)) + (28*(x0 // 14)) + (392*(x1 // 2)) + (784*r3) + (200704*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tl.where(rmask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask & xmask, tmp5_weight_next, tmp5_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp8 = tl.load(in_ptr0 + ((14*(x1 % 2)) + (28*(x0 // 14)) + (392*(x1 // 2)) + (784*r3) + (200704*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11 - tmp5
        tmp13 = 256.0
        tmp14 = tmp6 / tmp13
        tmp15 = 1e-06
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tl.store(out_ptr2 + (r3 + (256*x5)), tmp23, rmask & xmask)
''')
