

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/zq/czqxliilmixzt3upasvl3peq4jmhc6vrprrczgnqxiecc6lctsf6.py
# Source Nodes: [max_pool2d_6, pad_19], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# max_pool2d_6 => max_pool2d_with_indices_18
# pad_19 => constant_pad_nd_21
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_38 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_38', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_38', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 21
    x1 = (xindex // 432) % 21
    x0 = xindex % 432
    x5 = (xindex // 9072)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2*x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (864*x1) + (36288*x5)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, float("-inf"))
    tmp8 = 1 + (2*x1)
    tmp9 = tmp8 < tmp1
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (432 + x0 + (864*x1) + (36288*x5)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = triton_helpers.maximum(tmp12, tmp7)
    tmp14 = 2 + (2*x1)
    tmp15 = tmp14 < tmp1
    tmp16 = tmp2 & tmp15
    tmp17 = tl.load(in_ptr0 + (864 + x0 + (864*x1) + (36288*x5)), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.where(tmp16, tmp17, float("-inf"))
    tmp19 = triton_helpers.maximum(tmp18, tmp13)
    tmp20 = 1 + (2*x2)
    tmp21 = tmp20 < tmp1
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr0 + (18144 + x0 + (864*x1) + (36288*x5)), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp22, tmp23, float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp19)
    tmp26 = tmp21 & tmp9
    tmp27 = tl.load(in_ptr0 + (18576 + x0 + (864*x1) + (36288*x5)), tmp26 & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.where(tmp26, tmp27, float("-inf"))
    tmp29 = triton_helpers.maximum(tmp28, tmp25)
    tmp30 = tmp21 & tmp15
    tmp31 = tl.load(in_ptr0 + (19008 + x0 + (864*x1) + (36288*x5)), tmp30 & xmask, other=0.0).to(tl.float32)
    tmp32 = tl.where(tmp30, tmp31, float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp29)
    tmp34 = 2 + (2*x2)
    tmp35 = tmp34 < tmp1
    tmp36 = tmp35 & tmp4
    tmp37 = tl.load(in_ptr0 + (36288 + x0 + (864*x1) + (36288*x5)), tmp36 & xmask, other=0.0).to(tl.float32)
    tmp38 = tl.where(tmp36, tmp37, float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp33)
    tmp40 = tmp35 & tmp9
    tmp41 = tl.load(in_ptr0 + (36720 + x0 + (864*x1) + (36288*x5)), tmp40 & xmask, other=0.0).to(tl.float32)
    tmp42 = tl.where(tmp40, tmp41, float("-inf"))
    tmp43 = triton_helpers.maximum(tmp42, tmp39)
    tmp44 = tmp35 & tmp15
    tmp45 = tl.load(in_ptr0 + (37152 + x0 + (864*x1) + (36288*x5)), tmp44 & xmask, other=0.0).to(tl.float32)
    tmp46 = tl.where(tmp44, tmp45, float("-inf"))
    tmp47 = triton_helpers.maximum(tmp46, tmp43)
    tl.store(out_ptr0 + (x6), tmp47, xmask)
''')
