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
    Data selection from expression string.
    """

    ####################################################################################################################
    # TOKENIZER                                                                                                        #
    ####################################################################################################################

    class Token:

        def __init__(self, _type: str, _value: str):

            self.type: str = _type
            self.value: str = _value

    ####################################################################################################################

    _TOKEN_REGEX = re.compile(
        r'(==|!=|<=|>=|<|>)'
        r'|'
        r'([&|])'
        r'|'
        r'(~)'
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
    def _tokenize(expression: str) -> typing.Generator[Token, typing.Any, typing.Any]:

        for comparison_op, boolean_op, not_op, grouping, number, column, blank in Selection._TOKEN_REGEX.findall(expression):

            if   comparison_op:
                yield Selection.Token('COMPARISON_OP', comparison_op)
            elif boolean_op:
                yield Selection.Token('BOOLEAN_OP', boolean_op)
            elif not_op:
                yield Selection.Token('NOT_OP', not_op)
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

    class UnaryOpNode:

        def __init__(self, _op: str, _right):

            self.op: str = _op
            self.right = _right

    ####################################################################################################################

    class BinaryOpNode:

        def __init__(self, _left, _op: str, _right):

            self.left = _left
            self.op: str = _op
            self.right = _right

    ####################################################################################################################

    class NumberNode:

        def __init__(self, _value: str):

            self.value: str = _value

    ####################################################################################################################

    class ColumnNode:

        def __init__(self, _value: str):

            self.value: str = _value

    ####################################################################################################################

    @staticmethod
    def _pick_token(token_list: typing.List[Token]) -> typing.Optional[Token]:

        if token_list:

            return token_list.pop(0)

        else:

            raise ValueError('Syntax error')

    ####################################################################################################################

    @staticmethod
    def _parse_term(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]:

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
    def _parse_not_op(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]:

        if token_list and token_list[0].type == 'NOT_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_term(token_list)

            return Selection.UnaryOpNode(op, right_node)

        return Selection._parse_term(token_list)

    ####################################################################################################################

    @staticmethod
    def _parse_comparison_op(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]:

        ################################################################################################################

        left_node = Selection._parse_not_op(token_list)

        ################################################################################################################

        while token_list and token_list[0].type == 'COMPARISON_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_not_op(token_list)

            left_node = Selection.BinaryOpNode(left_node, op, right_node)

        ################################################################################################################

        return left_node

    ####################################################################################################################

    @staticmethod
    def _parse_boolean_op(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]:

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
    def parse(expression: str) -> typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]:

        """
        Evaluates the specified expression and returns the associated Abstract Syntax Tree (AST).

        .. code-block:: ebnf

            expression       = boolean_expr ;

            boolean_expr     = comparison_expr { BOOLEAN_OP comparison_expr } ;

            comparison_expr  = not_expr { COMPARISON_OP not_expr } ;

            not_expr         = [ NOT_OP ] term ;

            term             = "(" boolean_expr ")"
                             | NUMBER
                             | COLUMN
                             ;

            COMPARISON_OP    = "==" | "!=" | "<=" | ">=" | "<" | ">" ;
            BOOLEAN_OP       = "&" | "|" ;
            NOT_OP           = "~" ;

        Parameters
        ----------
        expression : str
            The expression to be evaluated.

        Returns
        -------
        Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]
            The Abstract Syntax Tree (AST).
        """

        token_list = list(Selection._tokenize(expression))

        result = Selection._parse_boolean_op(token_list)

        if token_list:

            raise ValueError('Syntax error')

        return result

    ####################################################################################################################
    # SELECTION                                                                                                        #
    ####################################################################################################################

    @staticmethod
    def _evaluate(node: typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode], table: np.ndarray) -> typing.Union[np.ndarray, float]:

        ################################################################################################################
        # UNARY OP                                                                                                     #
        ################################################################################################################

        if isinstance(node, Selection.UnaryOpNode):

            right_value = Selection._evaluate(node.right, table)

            if node.op == '~':
                return ~right_value

        ################################################################################################################
        # BINARY OP                                                                                                    #
        ################################################################################################################

        if isinstance(node, Selection.BinaryOpNode):

            left_value = Selection._evaluate(node.left, table)
            right_value = Selection._evaluate(node.right, table)

            if   node.op == '==':
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
    def to_string(node: typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode], is_root: bool = True) -> str:

        """
        Converts the specified Abstract Syntax Tree (AST) into the associated expression.

        Parameters
        ----------
        node : typing.Union[UnaryOpNode, BinaryOpNode, NumberNode, ColumnNode]
            An Abstract Syntax Tree node.
        is_root : bool, default: True
            Internal, don't use.

        Returns
        -------
        str
            The reformatted expression.
        """

        ################################################################################################################
        # UNARY OP                                                                                                     #
        ################################################################################################################

        if isinstance(node, Selection.UnaryOpNode):

            right_expr = Selection.to_string(node.right, is_root = False)

            expr = f'{node.op}{right_expr}'

            if not is_root:

                expr = f'({expr})'

            return expr

        ################################################################################################################
        # BINARY OP                                                                                                    #
        ################################################################################################################

        if isinstance(node, Selection.BinaryOpNode):

            left_expr = Selection.to_string(node.left, is_root = False)
            right_expr = Selection.to_string(node.right, is_root = False)

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
    def filter_table(expression: str, table: np.ndarray) -> typing.Tuple[str, np.ndarray]:

        """
        Evaluates the specified expression and filters the table.

        Parameters
        ----------
        expression : str
            The expression to be evaluated.
        table : np.ndarray
            The table to be filtered with the expression.

        Returns
        -------
        str
            The reformatted expression.
        np.ndarray
            The generate the selection mask.
        """

        expression = expression.strip()

        if not expression:

            return expression, np.full_like(table, 1, dtype = bool)

        else:

            ast = Selection.parse(expression)

            return (
                Selection.to_string(ast),
                Selection._evaluate(ast, table),
            )

########################################################################################################################
