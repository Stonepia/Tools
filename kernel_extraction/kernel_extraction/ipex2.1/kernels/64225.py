

# Original file: ./hf_Bert___60.0/hf_Bert___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/e7/ce75yyaaj57qxit5ps6esky23b5tgqblhoyvz7jdsgkflnkibsy2.py
# Source Nodes: [gelu_12, l__self___cls_predictions_transform_layer_norm], Original ATen: [aten.gelu, aten.native_layer_norm]
# gelu_12 => add_100, convert_element_type_231, convert_element_type_232, erf_12, mul_87, mul_88, mul_89
# l__self___cls_predictions_transform_layer_norm => add_101, add_102, convert_element_type_233, convert_element_type_234, mul_90, mul_91, rsqrt_25, sub_38, var_mean_25
triton_per_fused_gelu_native_layer_norm_7 = async_compile.triton('triton_per_fused_gelu_native_layer_norm_7', '''
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_gelu_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 768, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp11 - tmp21
    tmp29 = 768.0
    tmp30 = tmp27 / tmp29
    tmp31 = 1e-12
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')
