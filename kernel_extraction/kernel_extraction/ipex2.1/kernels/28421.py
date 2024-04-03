

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/wk/cwkyrotadxlqzm7zbgrazp6bnyqfn6cgap2wkdv2h56btas76nl5.py
# Source Nodes: [einsum_1, l__self___transformer_dropout], Original ATen: [aten._to_copy, aten.native_dropout]
# einsum_1 => convert_element_type_3
# l__self___transformer_dropout => gt
triton_poi_fused__to_copy_native_dropout_1 = async_compile.triton('triton_poi_fused__to_copy_native_dropout_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_dropout_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_native_dropout_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 1024)
    x1 = xindex % 1024
    tmp6 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tl.where(tmp6 < 0, tmp6 + 32000, tmp6)
    # tl.device_assert((0 <= tmp7) & (tmp7 < 32000), "index out of bounds: 0 <= tmp7 < 32000")
    tmp8 = tl.load(in_ptr2 + (x1 + (1024*tmp7)), None)
    tmp9 = tmp5 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp12, None)
''')
