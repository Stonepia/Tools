

# Original file: ./speech_transformer__22_inference_62.2/speech_transformer__22_inference_62.2_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/oj/cojfgik4vrzkgcr3czsawsmja6vglo4xervkzk4ce7fplcgqcqlo.py
# Source Nodes: [add_1, imul, l__self___layer_stack_0_pos_ffn_w_1, l__self___layer_stack_0_slf_attn_dropout, l__self___layer_stack_0_slf_attn_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mul, aten.native_layer_norm]
# add_1 => add_3
# imul => mul_4
# l__self___layer_stack_0_pos_ffn_w_1 => convert_element_type_18
# l__self___layer_stack_0_slf_attn_dropout => clone_6
# l__self___layer_stack_0_slf_attn_layer_norm => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
triton_per_fused__to_copy_add_clone_mul_native_layer_norm_5 = async_compile.triton('triton_per_fused__to_copy_add_clone_mul_native_layer_norm_5', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mul_native_layer_norm_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mul_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 2040
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 512, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp3 - tmp13
    tmp21 = 512.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp33, rmask & xmask)
''')
