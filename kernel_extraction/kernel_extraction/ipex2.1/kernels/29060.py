

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/yv/cyv36vivzjpjevyhlcl2uui42h5ayv7tr72tbtpzwaxewh3hkytx.py
# Source Nodes: [add_10, add_11, add_12, add_13, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___mlp_drop2, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___mlp_drop2, getattr_getattr_l__mod___stages___3_____1___norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_10 => add_76
# add_11 => add_79
# add_12 => add_82
# add_13 => add_85
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_proj_drop => clone_38
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___mlp_drop2 => clone_40
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_proj_drop => clone_45
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___mlp_drop2 => clone_47
# getattr_getattr_l__mod___stages___3_____1___norm => convert_element_type_153, var_mean_13
triton_per_fused_add_clone_native_layer_norm_40 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_40', '''
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
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask, other=0.0).to(tl.float32)
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
    tmp17 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask)
    tl.store(out_ptr0 + (x0), tmp19, None)
    tl.store(out_ptr1 + (x0), tmp25, None)
''')
