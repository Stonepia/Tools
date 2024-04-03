

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/wv/cwv5lneqdoa45nofsff37p67jybeqtrepegejtowomprrndektgq.py
# Source Nodes: [dropout_8, float_11, softmax_8, type_as_8], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# dropout_8 => gt_36, mul_153, mul_154
# float_11 => convert_element_type_150
# softmax_8 => amax_8, div_12, exp_8, sub_13, sum_9
# type_as_8 => convert_element_type_151
triton_per_fused__softmax__to_copy_native_dropout_18 = async_compile.triton('triton_per_fused__softmax__to_copy_native_dropout_18', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_native_dropout_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_native_dropout_18(in_ptr0, in_ptr1, in_ptr2, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tmp2 = (-1)*(tl.minimum(0, r3 + ((-1)*x0), tl.PropagateNan.NONE))
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tmp2.to(tl.float32)
    tmp6 = 16.0
    tmp7 = tmp5 / tmp6
    tmp8 = tl.log(tmp7)
    tmp9 = 2.0794415416798357
    tmp10 = tmp8 / tmp9
    tmp11 = tmp10 * tmp6
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp12 + tmp3
    tmp14 = tl.full([1, 1], 31, tl.int64)
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp2, tmp15)
    tmp17 = tl.full([1, 1], 0, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(tmp18 < 0, tmp18 + 32, tmp18)
    # tl.device_assert((0 <= tmp19) & (tmp19 < 32), "index out of bounds: 0 <= tmp19 < 32")
    tmp20 = tl.load(in_ptr1 + (x1 + (6*tmp19)), None)
    tmp21 = r3
    tmp22 = x0
    tmp23 = tmp21 <= tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = -3.4028234663852886e+38
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 + tmp28
    tmp30 = tmp1 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask, tmp33, float("-inf"))
    tmp36 = triton_helpers.max2(tmp35, 1)[:, None]
    tmp37 = tmp32 - tmp36
    tmp38 = tl.exp(tmp37)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None]
    tmp43 = tl.load(in_ptr2 + load_seed_offset)
    tmp44 = r3 + (128*x4)
    tmp45 = tl.rand(tmp43, (tmp44).to(tl.uint32))
    tmp46 = tmp45.to(tl.float32)
    tmp47 = 0.1
    tmp48 = tmp46 > tmp47
    tmp49 = tmp38 / tmp42
    tmp50 = tmp48.to(tl.float32)
    tmp51 = tmp49.to(tl.float32)
    tmp52 = tmp50 * tmp51
    tmp53 = 1.1111111111111112
    tmp54 = tmp52 * tmp53
    tl.store(out_ptr4 + (r3 + (128*x4)), tmp48, rmask)
    tl.store(out_ptr5 + (r3 + (128*x4)), tmp49, rmask)
    tl.store(out_ptr6 + (r3 + (128*x4)), tmp54, rmask)
''')
