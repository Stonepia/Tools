

# Original file: ./detectron2_fcos_r_50_fpn__75_inference_115.55/detectron2_fcos_r_50_fpn__75_inference_115.55_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ns/cnsm53fzn5xevvqkokcyiq3plx5wfni7coovo3nppd4zfoudyvwt.py
# Source Nodes: [add, add_1, mul, mul_1, stack], Original ATen: [aten.add, aten.mul, aten.stack]
# add => add
# add_1 => add_1
# mul => mul
# mul_1 => mul_1
# stack => cat
triton_poi_fused_add_mul_stack_0 = async_compile.triton('triton_poi_fused_add_mul_stack_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_stack_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.where(tmp0 < 0, tmp0 + ks0, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < ks0)) | ~xmask, "index out of bounds: 0 <= tmp1 < ks0")
    tmp2 = tl.load(in_ptr1 + (2*tmp1), xmask)
    tmp3 = tl.where(tmp2 < 0, tmp2 + ks1, tmp2)
    # tl.device_assert(((0 <= tmp3) & (tmp3 < ks1)) | ~xmask, "index out of bounds: 0 <= tmp3 < ks1")
    tmp4 = tl.load(in_ptr2 + (4*tmp3), xmask)
    tmp5 = tl.load(in_ptr2 + (2 + (4*tmp3)), xmask)
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 - tmp4
    tmp10 = tl.load(in_ptr2 + (3 + (4*tmp3)), xmask)
    tmp11 = tl.load(in_ptr2 + (1 + (4*tmp3)), xmask)
    tmp12 = tmp10 - tmp11
    tmp13 = tmp11 + tmp10
    tmp14 = tmp13 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (4*x0), tmp9, xmask)
    tl.store(out_ptr2 + (4*x0), tmp9, xmask)
    tl.store(out_ptr3 + (4*x0), tmp12, xmask)
    tl.store(out_ptr4 + (4*x0), tmp12, xmask)
    tl.store(out_ptr5 + (x0), tmp14, xmask)
''')