

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/rl/crllrjs5e4kvqo5mjkylta3qzvp3omjxxnrfxnaa6aqxecikbqda.py
# Source Nodes: [add_2, add_3, add_4, add_5, getattr_getattr_getattr_l__self___stages___2_____1___transformer___0___attn_proj_drop, getattr_getattr_getattr_l__self___stages___2_____1___transformer___0___mlp_drop2, getattr_getattr_getattr_l__self___stages___2_____1___transformer___1___attn_proj_drop, getattr_getattr_getattr_l__self___stages___2_____1___transformer___1___mlp_drop2, getattr_getattr_l__self___stages___2_____1___norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_2 => add_38
# add_3 => add_41
# add_4 => add_44
# add_5 => add_47
# getattr_getattr_getattr_l__self___stages___2_____1___transformer___0___attn_proj_drop => clone_6
# getattr_getattr_getattr_l__self___stages___2_____1___transformer___0___mlp_drop2 => clone_8
# getattr_getattr_getattr_l__self___stages___2_____1___transformer___1___attn_proj_drop => clone_13
# getattr_getattr_getattr_l__self___stages___2_____1___transformer___1___mlp_drop2 => clone_15
# getattr_getattr_l__self___stages___2_____1___norm => convert_element_type_126, var_mean_4
triton_per_fused_add_clone_native_layer_norm_18 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_18', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + ((144*(((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + ((x1 % 4) % 2)) % 32)) + (4608*((((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 147456)) + ((x1 % 4) % 2)) // 32) % 32)) + (147456*((((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 147456)) + ((x1 % 4) % 2)) // 147456) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 147456)) + ((x1 % 4) % 2)) // 1024) % 144)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (144*x3)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r2 + (144*x3)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r2 + (144*x3)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (144*x3)), rmask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tl.store(in_out_ptr0 + (r2 + (144*x3)), tmp8, rmask)
    tl.store(out_ptr0 + (x3), tmp19, None)
    tl.store(out_ptr1 + (x3), tmp25, None)
''')
