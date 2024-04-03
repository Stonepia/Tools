

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/7a/c7ak5j4lyhpnz2eqdxbyvozeuff6z7i366h6wl3peix3g7vwab2q.py
# Source Nodes: [add_73, add_74, l__self___cls_predictions_decoder, l__self___cls_predictions_transform_layer_norm, mul_276, mul_277, mul_278, mul_279, pow_13, tanh_12], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh]
# add_73 => add_256
# add_74 => add_257
# l__self___cls_predictions_decoder => convert_element_type_463
# l__self___cls_predictions_transform_layer_norm => add_258, add_259, mul_343, mul_344, rsqrt_25, sub_181, var_mean_25
# mul_276 => mul_339
# mul_277 => mul_340
# mul_278 => mul_341
# mul_279 => mul_342
# pow_13 => convert_element_type_462, pow_13
# tanh_12 => tanh_13
triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_32 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_32', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_32(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0.to(tl.float32)
    tmp5 = tmp4 * tmp4
    tmp6 = tmp5 * tmp4
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp3 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 768.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-12
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp43, rmask)
''')
