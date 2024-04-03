

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/wb/cwbvgl6hwkrseyxybngczqf4gz2avqynphih5a2rpyiuu344nj5z.py
# Source Nodes: [add_64, add_67, add_70, add_73, l__mod___transformer_blocks_10_input_sublayer_dropout, l__mod___transformer_blocks_10_output_sublayer_dropout, l__mod___transformer_blocks_11_input_sublayer_dropout, l__mod___transformer_blocks_11_output_sublayer_dropout], Original ATen: [aten.add, aten.clone]
# add_64 => add_74
# add_67 => add_78
# add_70 => add_81
# add_73 => add_85
# l__mod___transformer_blocks_10_input_sublayer_dropout => clone_96
# l__mod___transformer_blocks_10_output_sublayer_dropout => clone_98
# l__mod___transformer_blocks_11_input_sublayer_dropout => clone_105
# l__mod___transformer_blocks_11_output_sublayer_dropout => clone_107
triton_poi_fused_add_clone_11 = async_compile.triton('triton_poi_fused_add_clone_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')
