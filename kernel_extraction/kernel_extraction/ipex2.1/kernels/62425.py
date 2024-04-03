

# Original file: ./basic_gnn_gcn__22_inference_62.2/basic_gnn_gcn__22_inference_62.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/m4/cm4xldthqbwzg2iaiotpsuujl7wm5jsekxg6uzfyynv262jby3sw.py
# Source Nodes: [eq, getitem_3, masked_fill_, mul, mul_1, pow_], Original ATen: [aten.eq, aten.index, aten.masked_fill, aten.mul, aten.pow]
# eq => eq
# getitem_3 => index_1
# masked_fill_ => full_default_2, where
# mul => index
# mul_1 => mul_1
# pow_ => pow_1
triton_poi_fused_eq_index_masked_fill_mul_pow_2 = async_compile.triton('triton_poi_fused_eq_index_masked_fill_mul_pow_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_index_masked_fill_mul_pow_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_eq_index_masked_fill_mul_pow_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 209993
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp9 = tl.load(in_ptr0 + (209993 + x0), xmask)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 10000, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 10000)) | ~xmask, "index out of bounds: 0 <= tmp1 < 10000")
    tmp2 = tl.load(in_ptr1 + (tmp1), xmask)
    tmp3 = -0.5
    tmp4 = libdevice.pow(tmp2, tmp3)
    tmp5 = float("inf")
    tmp6 = tmp4 == tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp4)
    tmp10 = tl.where(tmp9 < 0, tmp9 + 10000, tmp9)
    # tl.device_assert(((0 <= tmp10) & (tmp10 < 10000)) | ~xmask, "index out of bounds: 0 <= tmp10 < 10000")
    tmp11 = tl.load(in_ptr1 + (tmp10), xmask)
    tmp12 = libdevice.pow(tmp11, tmp3)
    tmp13 = tmp12 == tmp5
    tmp14 = tl.where(tmp13, tmp7, tmp12)
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')
