

# Original file: ./cm3leon_generate__26_inference_66.6/cm3leon_generate__26_inference_66.6.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/zl/czlxdli3udfjhz2jomqx65hf3qpvy54ztgvmzxtrx2x6c5h2vrcy.py
# Source Nodes: [iadd_12, nan_to_num, nan_to_num_11, softmax_11, type_as_13], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_12 => add_81
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_11 => convert_element_type_94, eq_22, eq_23, isnan_11, where_34, where_35, where_36
# softmax_11 => amax_11, convert_element_type_96, div_11, exp_11, sub_35, sum_12
# type_as_13 => convert_element_type_97
triton_per_fused__softmax__to_copy_add_nan_to_num_45 = async_compile.triton('triton_per_fused__softmax__to_copy_add_nan_to_num_45', '''
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
    size_hints=[16, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_nan_to_num_45', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_nan_to_num_45(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp2 = float("inf")
    tmp3 = tmp1 == tmp2
    tmp4 = float("-inf")
    tmp5 = tmp1 == tmp4
    tmp6 = libdevice.isnan(tmp0).to(tl.int1)
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp0)
    tmp9 = -3.3895313892515355e+38
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = 3.3895313892515355e+38
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, float("-inf"))
    tmp20 = triton_helpers.max2(tmp19, 1)[:, None]
    tmp21 = tmp16 - tmp20
    tmp22 = tl.exp(tmp21)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp22 / tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (36*x0)), tmp28, rmask & xmask)
''')