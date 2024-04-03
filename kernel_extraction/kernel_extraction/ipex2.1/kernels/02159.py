

# Original file: ./OPTForCausalLM__21_forward_62.1/OPTForCausalLM__21_forward_62.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/t4/ct4q76nalqpznmcx4zmumbl64z2r3kuetewb4bbdz42hdxqkjcvs.py
# Source Nodes: [add, getitem, long, mul, sub], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.slice, aten.sub]
# add => add
# getitem => slice_1, slice_2
# long => convert_element_type
# mul => mul
# sub => sub
triton_poi_fused__to_copy_add_mul_slice_sub_1 = async_compile.triton('triton_poi_fused__to_copy_add_mul_slice_sub_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_slice_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_slice_sub_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = tmp0 * tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 + tmp6
    tl.store(in_out_ptr0 + (x0), tmp7, None)
''')
