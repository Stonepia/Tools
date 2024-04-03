

# Original file: ./pit_b_224___60.0/pit_b_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/nj/cnjk5hrgncleoybagdwbqgce36ir52o3i6mob7mv746hb4pdejkg.py
# Source Nodes: [l__mod___head_drop, l__mod___norm], Original ATen: [aten.clone, aten.native_layer_norm]
# l__mod___head_drop => clone_41
# l__mod___norm => clone_40, convert_element_type_78, var_mean_26
triton_per_fused_clone_native_layer_norm_36 = async_compile.triton('triton_per_fused_clone_native_layer_norm_36', '''
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
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_36', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_clone_native_layer_norm_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (66560*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (66560*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (66560*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (66560*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (66560*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 1024, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 1024.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-06
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp39, rmask & xmask)
''')