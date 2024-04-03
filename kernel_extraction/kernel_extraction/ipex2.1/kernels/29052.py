

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/ma/cmagmyrfdiwnsgqtg2kngrobrib647m77ei5jopopnenbbky5iui.py
# Source Nodes: [add_6, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_6 => add_64
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_proj_drop => clone_24
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => add_65, add_66, convert_element_type_125, convert_element_type_126, mul_106, mul_107, rsqrt_6, sub_32, var_mean_6
triton_red_fused_add_clone_native_layer_norm_32 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_32', '''
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
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((192*(((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + ((x1 % 4) % 2)) % 16)) + (3072*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 16) % 16)) + (49152*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 49152) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 256) % 192)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (192*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tl.where(rmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask, tmp5_weight_next, tmp5_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr0 + ((192*(((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + ((x1 % 4) % 2)) % 16)) + (3072*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 16) % 16)) + (49152*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 49152) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 256) % 192)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r2 + (192*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 - tmp5
        tmp13 = 192.0
        tmp14 = tmp6 / tmp13
        tmp15 = 1e-05
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (192*x3)), tmp25, rmask)
''')
