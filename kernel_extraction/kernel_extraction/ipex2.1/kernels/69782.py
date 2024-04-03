

# Original file: ./torch_multimodal_clip___60.0/torch_multimodal_clip___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/pe/cpepz4b4fwpno7vrflcncjqze6xc6uros3xdup2og7kzeamiezcb.py
# Source Nodes: [add, l__self___encoder_a_encoder, layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
# add => add
# l__self___encoder_a_encoder => convert_element_type_3, var_mean_1
# layer_norm => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_per_fused__to_copy_add_native_layer_norm_3 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_3', '''
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr5, xnumel, rnumel):
    xnumel = 1600
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 50
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp36 / tmp11
    tmp38 = tmp30 - tmp37
    tmp39 = tmp38 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [RBLOCK])
    tmp42 = tl.where(rmask & xmask, tmp40, 0)
    tmp43 = triton_helpers.promote_to_tensor(tl.sum(tmp42, 0))
    tmp44 = tmp29 - tmp37
    tmp45 = tmp43 / tmp20
    tmp46 = tmp45 + tmp22
    tmp47 = libdevice.rsqrt(tmp46)
    tmp48 = tmp44 * tmp47
    tmp49 = tmp48 * tmp26
    tmp50 = tmp49 + tmp28
    tmp51 = tmp50.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp29, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (768*x3)), tmp51, rmask & xmask)
''')
