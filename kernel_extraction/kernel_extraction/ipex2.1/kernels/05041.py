

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/z6/cz6aszokddtjeuv4hucq2wibt4oyty3i3ujrgir5ko4y46dtyelq.py
# Source Nodes: [add_20, contiguous_24, getattr_getattr_l__mod___layers___2___blocks___1___mlp_drop2, getattr_getattr_l__mod___layers___2___blocks___2___norm1], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_20 => add_56
# contiguous_24 => clone_69
# getattr_getattr_l__mod___layers___2___blocks___1___mlp_drop2 => clone_68
# getattr_getattr_l__mod___layers___2___blocks___2___norm1 => var_mean_15
triton_per_fused_add_clone_native_layer_norm_41 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_41', '''
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
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_41', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 12544
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    x4 = xindex % 7
    x5 = (xindex // 7) % 2
    x6 = (xindex // 14) % 7
    x7 = (xindex // 98)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*(x0 % 14)) + (7168*((((14*(x0 // 14)) + (x0 % 14)) // 14) % 14)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*(((((14*(x0 // 14)) + (x0 % 14)) // 14) % 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(((((14*(x0 // 14)) + (x0 % 14)) // 14) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x4) + (3584*x6) + (25088*x5) + (50176*x7)), tmp35, rmask & xmask)
''')