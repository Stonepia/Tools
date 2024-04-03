

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/um/cumm2lzf55met6ycp22fnzckvubidp33dujafliz4o7jcd7rle44.py
# Source Nodes: [add_2, add_3, add_7, l__self___blocks_0_mlp_in_drop2, l__self___blocks_1_mlp_in_fc1, l__self___blocks_1_norm_mlp_in], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_2 => add_10
# add_3 => add_14
# add_7 => add_27
# l__self___blocks_0_mlp_in_drop2 => clone_11
# l__self___blocks_1_mlp_in_fc1 => convert_element_type_52
# l__self___blocks_1_norm_mlp_in => add_28, add_29, clone_24, mul_25, mul_26, rsqrt_8, sub_11, var_mean_8
triton_per_fused__to_copy_add_clone_native_layer_norm_22 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_22', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr1 + (x0 + (16*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr2 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp14 - tmp24
    tmp32 = 24.0
    tmp33 = tmp30 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp41.to(tl.float32)
    tl.store(out_ptr0 + (r2 + (24*x3)), tmp14, rmask)
    tl.store(out_ptr3 + (r2 + (24*x3)), tmp42, rmask)
''')
