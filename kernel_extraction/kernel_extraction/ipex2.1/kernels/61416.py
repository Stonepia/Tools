

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/e2/ce2losskuf6dvcdycuqobvs3ssyujv2244i34lizxcctpyo22rk6.py
# Source Nodes: [dropout_8, float_11, softmax_8, type_as_8], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# dropout_8 => gt_36, mul_153, mul_154
# float_11 => convert_element_type_93
# softmax_8 => amax_8, div_12, exp_8, sub_13, sum_9
# type_as_8 => convert_element_type_94
triton_per_fused__softmax__to_copy_native_dropout_14 = async_compile.triton('triton_per_fused__softmax__to_copy_native_dropout_14', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_native_dropout_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_native_dropout_14(in_ptr0, in_ptr1, in_ptr2, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, other=0.0).to(tl.float32)
    tmp1 = (-1)*(tl.minimum(0, r3 + ((-1)*x0), tl.PropagateNan.NONE))
    tmp2 = tl.full([1, 1], 16, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tmp1.to(tl.float32)
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.log(tmp6)
    tmp8 = 2.0794415416798357
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9 * tmp5
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tmp11 + tmp2
    tmp13 = tl.full([1, 1], 31, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tl.where(tmp3, tmp1, tmp14)
    tmp16 = tl.full([1, 1], 0, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.where(tmp17 < 0, tmp17 + 32, tmp17)
    # tl.device_assert((0 <= tmp18) & (tmp18 < 32), "index out of bounds: 0 <= tmp18 < 32")
    tmp19 = tl.load(in_ptr1 + (x1 + (6*tmp18)), None).to(tl.float32)
    tmp20 = r3
    tmp21 = x0
    tmp22 = tmp20 <= tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = -65504.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp19 + tmp28
    tmp30 = tmp0 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask, tmp32, float("-inf"))
    tmp35 = triton_helpers.max2(tmp34, 1)[:, None]
    tmp36 = tmp31 - tmp35
    tmp37 = tl.exp(tmp36)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tl.load(in_ptr2 + load_seed_offset)
    tmp43 = r3 + (128*x4)
    tmp44 = tl.rand(tmp42, (tmp43).to(tl.uint32))
    tmp45 = tmp44.to(tl.float32)
    tmp46 = 0.1
    tmp47 = tmp45 > tmp46
    tmp48 = tmp37 / tmp41
    tmp49 = tmp47.to(tl.float32)
    tmp50 = tmp48.to(tl.float32)
    tmp51 = tmp49 * tmp50
    tmp52 = 1.1111111111111112
    tmp53 = tmp51 * tmp52
    tl.store(out_ptr4 + (r3 + (128*x4)), tmp47, rmask)
    tl.store(out_ptr5 + (r3 + (128*x4)), tmp48, rmask)
    tl.store(out_ptr6 + (r3 + (128*x4)), tmp53, rmask)
''')
