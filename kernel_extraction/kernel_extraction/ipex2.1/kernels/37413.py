

# Original file: ./RobertaForCausalLM__0_forward_169.0/RobertaForCausalLM__0_forward_169.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/4k/c4ka3t3bmtgu36d3k72xh3hll3cath366sx6vjh5efgoliki6lco.py
# Source Nodes: [gelu_12, l__self___lm_head_decoder, l__self___lm_head_layer_norm], Original ATen: [aten.gelu, aten.native_layer_norm, aten.view]
# gelu_12 => add_102, convert_element_type_234, convert_element_type_235, erf_12, mul_162, mul_163, mul_164
# l__self___lm_head_decoder => view_266
# l__self___lm_head_layer_norm => add_103, add_104, convert_element_type_236, convert_element_type_237, mul_165, mul_166, rsqrt_25, sub_38, var_mean_25
triton_per_fused_gelu_native_layer_norm_view_13 = async_compile.triton('triton_per_fused_gelu_native_layer_norm_view_13', '''
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
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_view_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_gelu_native_layer_norm_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 8192
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
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 768, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = 768.0
    tmp29 = tmp27 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp11 - tmp21
    tmp34 = tmp33 * tmp32
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp10, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask)
    tl.store(out_ptr1 + (x0), tmp21, None)
''')