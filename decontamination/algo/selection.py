# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import re
import typing

import numpy as np

########################################################################################################################

class Selection(object):

    """
    Data selection.
    """

    ####################################################################################################################
    # TOKENIZER                                                                                                        #
    ####################################################################################################################

    class Token:

        def __init__(self, type: str, value: str):

            self.type: str = type
            self.value: str = value

    ####################################################################################################################

    TOKEN_REGEX = re.compile(
        r'(==|!=|<=|>=|<|>)'
        r'|'
        r'([&|])'
        r'|'
        r'([()])'
        r'|'
        r'([-+]?(?:'
            r'\d+\.\d*(?:[eE][-+]?\d+)?'
            r'|'
            r'\.\d+(?:[eE][-+]?\d+)?'
            r'|'
            r'\d+(?:[eE][-+]?\d+)?'
        r'))'
        r'|'
        r'(\w+)'
        r'|'
        r'(\s+)'
    )

    ####################################################################################################################

    @staticmethod
    def tokenize(expression: str) -> typing.Generator[Token, typing.Any, typing.Any]:

        for comparison_op, boolean_op, grouping, number, column, blank in Selection.TOKEN_REGEX.findall(expression):

            if   comparison_op:
                yield Selection.Token('COMPARISON_OP', comparison_op)
            elif boolean_op:
                yield Selection.Token('BOOLEAN_OP', boolean_op)
            elif grouping:
                yield Selection.Token('GROUPING', grouping)
            elif number:
                yield Selection.Token('NUMBER', number)
            elif column:
                yield Selection.Token('COLUMN', column)
            elif blank:
                # IGNORE #
                pass
            else:
                raise ValueError('Invalid token')

    ####################################################################################################################
    # PARSER                                                                                                           #
    ####################################################################################################################

    class BinaryOpNode:

        def __init__(self, left, op, right):

            self.left = left
            self.op: str = op
            self.right = right

    ####################################################################################################################

    class NumberNode:

        def __init__(self, value):

            self.value: str = value

    ####################################################################################################################

    class ColumnNode:

        def __init__(self, value):

            self.value: str = value

    ####################################################################################################################

    @staticmethod
    def _pick_token(token_list: typing.List[Token]) -> typing.Optional[Token]:

        if token_list:

            return token_list.pop(0)

        else:

            raise ValueError('Syntax error')

    ####################################################################################################################

    @staticmethod
    def _parse_term(token_list: typing.List[Token]) -> typing.Union[BinaryOpNode, NumberNode, ColumnNode]:

        ################################################################################################################

        token = Selection._pick_token(token_list)

        ################################################################################################################
        # GROUPING                                                                                                     #
        ################################################################################################################

        if token.value == '(':

            node = Selection._parse_boolean_op(token_list)

            token = Selection._pick_token(token_list)

            if token.value != ')':

                raise ValueError('Syntax error')

            return node

        ################################################################################################################
        # NUMBER                                                                                                       #
        ################################################################################################################

        elif token.type == 'NUMBER':

            return Selection.NumberNode(token.value)

        ################################################################################################################
        # COLUMN                                                                                                       #
        ################################################################################################################

        elif token.type == 'COLUMN':

            return Selection.ColumnNode(token.value)

        ################################################################################################################

        raise ValueError('Syntax error')

    ####################################################################################################################

    @staticmethod
    def _parse_comparison_op(token_list: typing.List[Token]) -> typing.Union[BinaryOpNode, NumberNode, ColumnNode]:

        ################################################################################################################

        left_node = Selection._parse_term(token_list)

        ################################################################################################################

        while token_list and token_list[0].type == 'COMPARISON_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_term(token_list)

            left_node = Selection.BinaryOpNode(left_node, op, right_node)

        ################################################################################################################

        return left_node

    ####################################################################################################################

    @staticmethod
    def _parse_boolean_op(token_list: typing.List[Token]) -> typing.Union[BinaryOpNode, NumberNode, ColumnNode]:

        ################################################################################################################

        left_node = Selection._parse_comparison_op(token_list)

        ################################################################################################################

        while token_list and token_list[0].type == 'BOOLEAN_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_comparison_op(token_list)

            left_node = Selection.BinaryOpNode(left_node, op, right_node)

        ################################################################################################################

        return left_node

    ####################################################################################################################

    @staticmethod
    def parse(tokens: typing.Generator[Token, typing.Any, typing.Any]) -> typing.Union[BinaryOpNode, NumberNode, ColumnNode]:

        token_list = list(tokens)

        result = Selection._parse_boolean_op(token_list)

        if token_list:

            raise ValueError('Syntax error')

        return result

    ####################################################################################################################
    # SELECTION                                                                                                        #
    ####################################################################################################################

    @staticmethod
    def _evaluate(table: np.ndarray, node: typing.Union[BinaryOpNode, NumberNode, ColumnNode]) -> typing.Union[np.ndarray, float]:

        ################################################################################################################
        # BINARY OP                                                                                                    #
        ################################################################################################################

        if isinstance(node, Selection.BinaryOpNode):

            left_value = Selection._evaluate(table, node.left)
            right_value = Selection._evaluate(table, node.right)

            if node.op == '==':
                return left_value == right_value
            elif node.op == '!=':
                return left_value != right_value
            elif node.op == '<=':
                return left_value <= right_value
            elif node.op == '>=':
                return left_value >= right_value
            elif node.op == '<':
                return left_value < right_value
            elif node.op == '>':
                return left_value > right_value
            elif node.op == '&':
                return left_value & right_value
            elif node.op == '|':
                return left_value | right_value

        ################################################################################################################
        # NUMBER                                                                                                       #
        ################################################################################################################

        if isinstance(node, Selection.NumberNode):

            return float(node.value)

        ################################################################################################################
        # COLUMN                                                                                                       #
        ################################################################################################################

        if isinstance(node, Selection.ColumnNode):

            return table[node.value]

        ################################################################################################################

        raise ValueError('Internal error')

    ####################################################################################################################

    @staticmethod
    def stringify(node: typing.Union[BinaryOpNode, NumberNode, ColumnNode], is_root: bool = True) -> str:

        ################################################################################################################
        # BINARY OP                                                                                                    #
        ################################################################################################################

        if isinstance(node, Selection.BinaryOpNode):

            left_expr = Selection.stringify(node.left, is_root = False)
            right_expr = Selection.stringify(node.right, is_root = False)

            expr = f'{left_expr} {node.op} {right_expr}'

            if not is_root:

                expr = f'({expr})'

            return expr

        ################################################################################################################
        # NUMBER                                                                                                       #
        ################################################################################################################

        elif isinstance(node, Selection.NumberNode):

            return node.value

        ################################################################################################################
        # COLUMN                                                                                                       #
        ################################################################################################################

        elif isinstance(node, Selection.ColumnNode):

            return node.value

        ################################################################################################################

        raise ValueError('Internal error')

    ####################################################################################################################

    @staticmethod
    def evaluate(table: np.ndarray, expression: str) -> typing.Tuple[np.ndarray, str]:

        """
        Evaluates the specified expression and applies it to the table.

        Parameters
        ----------
        table : np.ndarray
            The table to be filtered.
        expression : str
            The expression to be evaluated.

        Returns
        -------
        np.ndarray
            The generate the selection mask.
        str
            The reformatted expression.
        """

        expression = expression.strip()

        if not expression:

            return np.full_like(table, 1, dtype = bool), expression

        else:

            ast = Selection.parse(Selection.tokenize(expression))

            mask = Selection._evaluate(table, ast)

            expression = Selection.stringify(ast)

            return mask, expression

########################################################################################################################
