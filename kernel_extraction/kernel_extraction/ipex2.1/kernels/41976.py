

# Original file: ./hf_distil_whisper___60.0/hf_distil_whisper___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/l4/cl4v5s6qgt7xales5y6pvdkpgikh3gs6dtfwt53jesatuhrpyxmb.py
# Source Nodes: [add, add_1, dropout_2, l__mod___encoder_layers_0_final_layer_norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add_2
# add_1 => add_5
# dropout_2 => clone_7
# l__mod___encoder_layers_0_final_layer_norm => add_6, add_7, clone_8, convert_element_type_8, convert_element_type_9, mul_10, mul_9, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_clone_native_layer_norm_8 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_8', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1500
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1500*r1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp3 * tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tl.full([1], 1024, tl.int32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 / tmp26
    tmp28 = tmp18 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tmp17 - tmp27
    tmp35 = 1024.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 * tmp42
    tmp44 = tmp1.to(tl.float32)
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp46, rmask & xmask)
''')
