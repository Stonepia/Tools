

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/mq/cmqy3sb3zzxs252hoyhwve7ldbwe74ot4odpt5ig7gx7gtfnycro.py
# Source Nodes: [add_12, add_13, add_17, add_8, l__mod___blocks_1_mlp_in_drop2, l__mod___blocks_2_mlp_in_drop2, l__mod___blocks_3_norm_mlp_in], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_12 => add_44
# add_13 => add_48
# add_17 => add_61
# add_8 => add_31
# l__mod___blocks_1_mlp_in_drop2 => clone_26
# l__mod___blocks_2_mlp_in_drop2 => clone_41
# l__mod___blocks_3_norm_mlp_in => add_62, add_63, clone_54, convert_element_type_64, convert_element_type_65, mul_61, mul_62, rsqrt_18, sub_25, var_mean_18
triton_per_fused_add_clone_native_layer_norm_29 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_29', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (24*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (24*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (24*x0)), rmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp17 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp9 - tmp19
    tmp27 = 24.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp39, rmask)
''')
