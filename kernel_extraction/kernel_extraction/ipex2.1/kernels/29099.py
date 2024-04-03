

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/7o/c7ol7osahuhp6py7wujlrvz3fuw6cwhrkt6raxdjv47npd3dbeef.py
# Source Nodes: [add_2, add_3, add_4, add_5, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___mlp_drop2, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___mlp_drop2, getattr_getattr_l__mod___stages___2_____1___norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_2 => add_38
# add_3 => add_41
# add_4 => add_44
# add_5 => add_47
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_proj_drop => clone_6
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___mlp_drop2 => clone_8
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_proj_drop => clone_13
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___mlp_drop2 => clone_15
# getattr_getattr_l__mod___stages___2_____1___norm => var_mean_4
triton_per_fused_add_clone_native_layer_norm_11 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((144*(((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + ((x1 % 4) % 2)) % 32)) + (4608*((((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 147456)) + ((x1 % 4) % 2)) // 32) % 32)) + (147456*((((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 147456)) + ((x1 % 4) % 2)) // 147456) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 147456)) + ((x1 % 4) % 2)) // 1024) % 144)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (144*x3)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (144*x3)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (144*x3)), rmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (144*x3)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(in_out_ptr0 + (r2 + (144*x3)), tmp8, rmask)
    tl.store(out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr1 + (x3), tmp24, None)
''')
