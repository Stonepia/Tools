

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/x6/cx62rppv4w7at4db2suqbtz2xign2rx2vpio7ut4uwtnt4qpjubu.py
# Source Nodes: [iadd_17, nan_to_num, nan_to_num_16, softmax_16, type_as_18], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_17 => add_116
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_16 => convert_element_type_134, eq_32, eq_33, isnan_16, where_49, where_50, where_51
# softmax_16 => amax_16, convert_element_type_136, div_16, exp_16, sub_50, sum_17
# type_as_18 => convert_element_type_137
triton_per_fused__softmax__to_copy_add_nan_to_num_64 = async_compile.triton('triton_per_fused__softmax__to_copy_add_nan_to_num_64', '''
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
    size_hints=[16, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_nan_to_num_64', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_nan_to_num_64(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 17
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (17*x0)), rmask & xmask, other=0.0).to(tl.float32)
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
    tmp9 = -65504.0
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = 65504.0
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
    tl.store(out_ptr2 + (r1 + (17*x0)), tmp28, rmask & xmask)
''')
