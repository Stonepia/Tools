

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/hz/chzrbio5ushzce6hcjmo654aln55qgbhwtwgtcfscvqhctukj2mk.py
# Source Nodes: [float_27, softmax_24, type_as_24], Original ATen: [aten._softmax, aten._to_copy]
# float_27 => convert_element_type_374
# softmax_24 => amax_24, div_28, exp_24, sub_29, sum_25
# type_as_24 => convert_element_type_375
triton_per_fused__softmax__to_copy_2 = async_compile.triton('triton_per_fused__softmax__to_copy_2', '''
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_2(in_ptr0, in_ptr1, out_ptr3, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = (-1)*(tl.minimum(0, r2 + ((-1)*x0), tl.PropagateNan.NONE))
    tmp3 = tl.full([1], 16, tl.int64)
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
    tmp14 = tl.full([1], 31, tl.int64)
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp2, tmp15)
    tmp17 = tl.full([1], 0, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(tmp18 < 0, tmp18 + 32, tmp18)
    # tl.device_assert((0 <= tmp19) & (tmp19 < 32), "index out of bounds: 0 <= tmp19 < 32")
    tmp20 = tl.load(in_ptr1 + (x1 + (16*tmp19)), None)
    tmp21 = r2
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
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, float("-inf"))
    tmp36 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp35, 0))
    tmp37 = tmp32 - tmp36
    tmp38 = tl.exp(tmp37)
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp43 = tmp38 / tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp44, rmask)
''')
