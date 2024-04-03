

# Original file: ./YituTechConvBert__0_backward_171.1/YituTechConvBert__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/5q/c5qidv23lxtvv6hy3lfs6eiiidzjqhtvroytohp6elbfsvsvalzn.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_37 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_37', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_37', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, out_ptr3, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0 + (512*r2) + (393216*x1)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr6 + (r2 + (768*x3)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (r2 + (768*x3)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 1.1111111111111112
    tmp39 = tmp37 * tmp38
    tmp40 = tmp35 * tmp39
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp15, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp34, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp40, rmask)
''')
