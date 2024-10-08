# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import re
import ast

########################################################################################################################

_CALL_RE = re.compile('(\\w+)_xpu\\s*\\(')

_JIT_X_RE = re.compile('(?:\\w*\\.)?jit\\.')

_CPU_CODE_RE = re.compile(re.escape('!--BEGIN-CPU--') + '.*?' + re.escape('!--END-CPU--'), flags = re.DOTALL)

_GPU_CODE_RE = re.compile(re.escape('!--BEGIN-GPU--') + '.*?' + re.escape('!--END-GPU--'), flags = re.DOTALL)

########################################################################################################################

class Preprocessor(ast.NodeTransformer):

    ####################################################################################################################

    def __init__(self, is_gpu: bool):

        self.is_gpu = is_gpu

    ####################################################################################################################

    def visit_If(self, node):

        ################################################################################################################

        test_is_gpu = isinstance(node.test, ast.Attribute) and node.test.attr == 'is_gpu'

        test_not_gpu = (
                isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                and
                isinstance(node.test.operand, ast.Attribute) and node.test.operand.attr == 'is_gpu'
        )

        ################################################################################################################

        if test_is_gpu:
            if self.is_gpu:
                return ast.If(test = ast.Constant(value = True), body = node.body, orelse = [])
            else:
                return ast.If(test = ast.Constant(value = True), body = node.orelse, orelse = [])
        elif test_not_gpu:
            if not self.is_gpu:
                return ast.If(test = ast.Constant(value = True), body = node.body, orelse = [])
            else:
                return ast.If(test = ast.Constant(value = True), body = node.orelse, orelse = [])

        ################################################################################################################

        return self.generic_visit(node)

    ####################################################################################################################

    def process(self, source_code):

        ################################################################################################################

        if self.is_gpu:
            source_code = _CALL_RE.sub(lambda m: f'jit_module.{m.group(1)}_gpu(', _JIT_X_RE.sub('jit.', _CPU_CODE_RE.sub('', source_code)))
        else:
            source_code = _CALL_RE.sub(lambda m: f'jit_module.{m.group(1)}_cpu(', _JIT_X_RE.sub('jit.', _GPU_CODE_RE.sub('', source_code)))

        ################################################################################################################

        source_code = ast.unparse(self.visit(ast.parse(source_code)))

        ################################################################################################################

        if self.is_gpu:

            return (
                source_code
                .replace('jit.is_gpu', 'True')
                .replace('jit.grid', 'cuda_module.grid')
                .replace('jit.local_empty', 'cuda_module.local.array')
                .replace('jit.shared_empty', 'cuda_module.shared.array')
                .replace('jit.syncthreads', 'cuda_module.syncthreads')
                .replace('jit.atomic_add', 'cuda_module.atomic.add')
                .replace('jit.atomic_sub', 'cuda_module.atomic.sub')
            )

        else:

            return (
                source_code
                .replace('jit.is_gpu', 'False')
                .replace('jit.grid', '# jit.grid')
                .replace('jit.local_empty', 'np.empty')
                .replace('jit.shared_empty', 'np.empty')
                .replace('jit.syncthreads', '# jit.syncthreads')
                .replace('jit.atomic_add', 'jit_module.atomic.add')
                .replace('jit.atomic_sub', 'jit_module.atomic.sub')
            )

########################################################################################################################
