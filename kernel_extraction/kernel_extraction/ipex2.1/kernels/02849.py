

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/xw/cxwg4yh4hm4ybiqpbrtakhya6cxb22vhejscjtjke3lapfm67n3x.py
# Source Nodes: [iadd_2, nan_to_num, nan_to_num_1, softmax_1, triu], Original ATen: [aten._softmax, aten.add, aten.nan_to_num, aten.triu]
# iadd_2 => add_11
# nan_to_num => full_default_3, full_default_4
# nan_to_num_1 => eq_2, eq_3, isnan_1, where_4, where_5, where_6
# softmax_1 => amax_1, sub_5
# triu => full_default_1
triton_poi_fused__softmax_add_nan_to_num_triu_10 = async_compile.triton('triton_poi_fused__softmax_add_nan_to_num_triu_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_nan_to_num_triu_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_nan_to_num_triu_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (2*x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (1 + (2*x1)), xmask, eviction_policy='evict_last')
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
    tmp24 = tmp23 == tmp1
    tmp25 = tmp23 == tmp3
    tmp26 = libdevice.isnan(tmp23).to(tl.int1)
    tmp27 = tl.where(tmp26, tmp6, tmp23)
    tmp28 = tl.where(tmp25, tmp8, tmp27)
    tmp29 = tl.where(tmp24, tmp10, tmp28)
    tmp30 = tmp29 + tmp13
    tmp31 = triton_helpers.maximum(tmp22, tmp30)
    tmp32 = tmp14 - tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''')
