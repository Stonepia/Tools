

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/gk/cgkqwdqzokbud3fure6nb7d6rtjhysx3wssmctciztc34pjbl7m6.py
# Source Nodes: [iadd_3, nan_to_num, nan_to_num_2, softmax_2, triu], Original ATen: [aten._softmax, aten.add, aten.nan_to_num, aten.triu]
# iadd_3 => add_18
# nan_to_num => full_default_3, full_default_4
# nan_to_num_2 => eq_4, eq_5, isnan_2, where_7, where_8, where_9
# softmax_2 => amax_2, exp_2, sub_8, sum_3
# triu => full_default_1
triton_poi_fused__softmax_add_nan_to_num_triu_16 = async_compile.triton('triton_poi_fused__softmax_add_nan_to_num_triu_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_nan_to_num_triu_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_nan_to_num_triu_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x0), xmask)
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (1 + (3*x0)), xmask)
    tmp24 = tl.load(in_ptr0 + (2 + (3*x0)), xmask)
    tmp1 = float("inf")
    tmp2 = tmp0 == tmp1
    tmp3 = float("-inf")
    tmp4 = tmp0 == tmp3
    tmp5 = libdevice.isnan(tmp0).to(tl.int1)
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = -3.4028234663852886e+38
    tmp9 = tl.where(tmp4, tmp8, tmp7)
    tmp10 = 3.4028234663852886e+38
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp14 = tmp11 + tmp13
    tmp16 = tmp15 == tmp1
    tmp17 = tmp15 == tmp3
    tmp18 = libdevice.isnan(tmp15).to(tl.int1)
    tmp19 = tl.where(tmp18, tmp6, tmp15)
    tmp20 = tl.where(tmp17, tmp8, tmp19)
    tmp21 = tl.where(tmp16, tmp10, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = triton_helpers.maximum(tmp14, tmp22)
    tmp25 = tmp24 == tmp1
    tmp26 = tmp24 == tmp3
    tmp27 = libdevice.isnan(tmp24).to(tl.int1)
    tmp28 = tl.where(tmp27, tmp6, tmp24)
    tmp29 = tl.where(tmp26, tmp8, tmp28)
    tmp30 = tl.where(tmp25, tmp10, tmp29)
    tmp31 = tmp30 + tmp13
    tmp32 = triton_helpers.maximum(tmp23, tmp31)
    tmp33 = tmp14 - tmp32
    tmp34 = tl.exp(tmp33)
    tmp35 = tmp22 - tmp32
    tmp36 = tl.exp(tmp35)
    tmp37 = tmp34 + tmp36
    tmp38 = tmp31 - tmp32
    tmp39 = tl.exp(tmp38)
    tmp40 = tmp37 + tmp39
    tl.store(out_ptr0 + (x0), tmp32, xmask)
    tl.store(out_ptr1 + (x0), tmp40, xmask)
''')
