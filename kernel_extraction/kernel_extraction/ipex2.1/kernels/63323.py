

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/td/ctdxa5y3z2voa33zu74ed4ukk2ospsa3u264wer6lkagzvvbevii.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten._to_copy, aten.add, aten.clone, aten.copy, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_19 = async_compile.triton('triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_19', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp32', 4: '*fp16', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data__to_copy_add_clone_copy_native_dropout_backward_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (r1 + (128*x0)), rmask)
    tmp22 = tl.load(in_ptr5 + (r1 + (128*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr7 + (r1 + (128*x0)), rmask)
    tmp34 = tl.load(in_ptr8 + (r1 + (128*x0)), rmask, other=0.0)
    tmp36 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp7 * tmp12
    tmp14 = tmp8 - tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp16 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp25 = tmp22 * tmp24
    tmp26 = tmp23 - tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp3
    tmp32 = tmp28 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp37 = tmp34 * tmp36
    tmp38 = tmp35 - tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp27 + tmp39
    tmp41 = tmp40 + tmp15
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp15, rmask)
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp41, rmask)
''')
