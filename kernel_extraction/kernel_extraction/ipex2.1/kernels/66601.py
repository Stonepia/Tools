

# Original file: ./M2M100ForConditionalGeneration__81_backward_336.32/M2M100ForConditionalGeneration__81_backward_336.32.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/nk/cnk4heihoignivzgksggu5ctbjhhe37ix7vey43lnmszkxhbuym5.py
# Source Nodes: [add_1, add_2, l__self___final_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_1 => add_3
# add_2 => add_6
# l__self___final_layer_norm => mul_14, sub_4
triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_2 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_2', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp34 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp14 * tmp10
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 1024.0
    tmp26 = tmp9 / tmp25
    tmp27 = tmp14 * tmp25
    tmp28 = tmp27 - tmp18
    tmp29 = tmp10 * tmp23
    tmp30 = tmp28 - tmp29
    tmp31 = tmp26 * tmp30
    tmp32 = tmp24 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = 1.1111111111111112
    tmp37 = tmp35 * tmp36
    tmp38 = tmp33 * tmp37
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp10, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp32, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp38, rmask)
''')
