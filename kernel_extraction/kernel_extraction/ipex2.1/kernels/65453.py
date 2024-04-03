

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/yk/cykemowqwvhpphz3t6xtu5f3iobcc6qdnvfmlrr6gh4skkqy2zk5.py
# Source Nodes: [add_2, add_3, add_7, l__mod___blocks_0_mlp_in_drop2, l__mod___blocks_1_norm_mlp_in], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_2 => add_10
# add_3 => add_14
# add_7 => add_27
# l__mod___blocks_0_mlp_in_drop2 => clone_11
# l__mod___blocks_1_norm_mlp_in => add_28, add_29, clone_24, convert_element_type_28, convert_element_type_29, mul_25, mul_26, rsqrt_8, sub_11, var_mean_8
triton_per_fused_add_clone_native_layer_norm_22 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_22', '''
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
    size_hints=[524288, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((24*(x0 % 4)) + (24*(tl.where(((4*((x1 % 196) % 14)) + (x0 % 4)) >= 0, 0, 56))) + (96*((x1 % 196) % 14)) + (1344*((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) + (1344*(tl.where(((4*((x1 % 196) // 14)) + ((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) >= 0, 0, 56))) + (5376*((x1 % 196) // 14)) + (75264*(x1 // 196)) + ((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0 + (16*r2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_out_ptr0 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tmp11 - tmp21
    tmp29 = 24.0
    tmp30 = tmp27 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp34 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(in_out_ptr0 + (r2 + (24*x3)), tmp10, rmask)
    tl.store(out_ptr2 + (r2 + (24*x3)), tmp41, rmask)
''')
