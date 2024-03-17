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

    class Token(object):

        def __init__(self, _type: str, _value: str):

            self.type: str = _type
            self.value: str = _value

    ####################################################################################################################

    _TOKEN_PATTERN = re.compile(
        r'([()])'
        r'|'
        r'(~)'
        r'|'
        r'(==|!=|<=|>=|<|>)'
        r'|'
        r'(&{2}|and)|(\|{2}|or)'
        r'|'
        r'(&)|(\|)'
        r'|'
        r'([-+]?(?:'
            r'\d+\.\d*(?:[eE][-+]?\d+)?'
            r'|'
            r'\.\d+(?:[eE][-+]?\d+)?'
            r'|'
            r'\d+[eE][-+]?\d+'
        r'))'
        r'|'
        r'([-+]?(?:'
            r'0x[0-9a-fA-F]+'
            r'|'
            r'Ob[01]+'
            r'|'
            r'\d+'
        r'))'
        r'|'
        r'(\w+)'
        r'|'
        r'(\s+)'
    )

    ####################################################################################################################

    @staticmethod
    def _tokenize(expression: str) -> typing.Generator[Token, typing.Any, typing.Any]:

        for grouping, not_op, comparison_op, logical_and_op, logical_or_op, bitwise_and_op, bitwise_or_op, float_num, int_num, col_name, blank in Selection._TOKEN_PATTERN.findall(expression):

            if grouping:
                yield Selection.Token('GROUPING', grouping)
            elif not_op:
                yield Selection.Token('NOT_OP', not_op)
            elif comparison_op:
                yield Selection.Token('COMPARISON_OP', comparison_op)
            elif logical_and_op:
                yield Selection.Token('LOGICAL_AND_OP', logical_and_op)
            elif logical_or_op:
                yield Selection.Token('LOGICAL_OR_OP', logical_or_op)
            elif bitwise_and_op:
                yield Selection.Token('BITWISE_AND_OP', bitwise_and_op)
            elif bitwise_or_op:
                yield Selection.Token('BITWISE_OR_OP', bitwise_or_op)
            elif float_num:
                yield Selection.Token('FLOAT_NUM', float_num)
            elif int_num:
                yield Selection.Token('INT_NUM', int_num)
            elif col_name:
                yield Selection.Token('COL_NAME', col_name)
            elif blank:
                # IGNORE #
                pass
            else:
                raise ValueError('Invalid token')

    ####################################################################################################################
    # PARSER                                                                                                           #
    ####################################################################################################################

    class UnaryOpNode(object):

        """An unary operator node"""

        def __init__(self, _op: str, _right):

            self.op: str = _op
            self.right = _right

    ####################################################################################################################

    class BinaryOpNode(object):

        """A binary operator node"""

        def __init__(self, _left, _op: str, _right):

            self.left = _left
            self.op: str = _op
            self.right = _right

    ####################################################################################################################

    class FloatNumNode(object):

        """A floating number node"""

        def __init__(self, _value: str):

            self.value: str = _value

    ####################################################################################################################

    class IntNumNode(object):

        """An integer number node"""

        def __init__(self, _value: str):

            self.value: str = _value

    ####################################################################################################################

    class ColNameNode(object):

        """A column name node"""

        def __init__(self, _value: str):

            self.value: str = _value

    ####################################################################################################################

    @staticmethod
    def _pick_token(
        token_list: typing.List[Token],
        expected_type: typing.Optional[str] = None,
        expected_value: typing.Optional[str] = None,
    ) -> typing.Optional[Token]:

        if not token_list:

            raise ValueError('Syntax error: truncated expression')

        if expected_type and token_list[0].type != expected_type:

            raise ValueError(f'Syntax error: expected token type `{expected_type}` but `{token_list[0].type}` found')

        if expected_value and token_list[0].value != expected_value:

            raise ValueError(f'Syntax error: expected token value `{expected_value}` but `{token_list[0].value}` found')

        return token_list.pop(0)

    ####################################################################################################################

    @staticmethod
    def _parse_term(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

        ################################################################################################################

        token = Selection._pick_token(token_list)

        ################################################################################################################
        # GROUPING                                                                                                     #
        ################################################################################################################

        if token.value == '(':

            node = Selection._parse_logical_or_expr(token_list)

            Selection._pick_token(token_list, expected_value = ')')

            return node

        ################################################################################################################
        # FLOAT_NUM                                                                                                    #
        ################################################################################################################

        elif token.type == 'FLOAT_NUM':

            return Selection.FloatNumNode(token.value)

        ################################################################################################################
        # INT_NUM                                                                                                      #
        ################################################################################################################

        elif token.type == 'INT_NUM':

            return Selection.IntNumNode(token.value)

        ################################################################################################################
        # COL_NAME                                                                                                     #
        ################################################################################################################

        elif token.type == 'COL_NAME':

            ############################################################################################################

            left_node = Selection.ColNameNode(token.value)

            ############################################################################################################

            if token_list and token_list[0].type in ['LOGICAL_AND_OP', 'LOGICAL_OR_OP']:

                op = token_list.pop(0).value

                token = Selection._pick_token(token_list, expected_type = 'INT_NUM')

                right_node = Selection.FloatNumNode(token.value)

                left_node = Selection.BinaryOpNode(left_node, op, right_node)

            ############################################################################################################

            return left_node

        ################################################################################################################

        raise ValueError(f'Syntax error: unexpected token type `{token.type}`')

    ####################################################################################################################

    @staticmethod
    def _parse_not_op(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

        if token_list and token_list[0].type == 'NOT_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_term(token_list)

            return Selection.UnaryOpNode(op, right_node)

        return Selection._parse_term(token_list)

    ####################################################################################################################

    @staticmethod
    def _parse_comparison_op(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

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
    def _parse_logical_and_expr(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

        ################################################################################################################

        left_node = Selection._parse_comparison_op(token_list)

        ################################################################################################################

        while token_list and token_list[0].type == 'LOGICAL_AND_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_comparison_op(token_list)

            left_node = Selection.BinaryOpNode(left_node, op, right_node)

        ################################################################################################################

        return left_node

    ####################################################################################################################

    @staticmethod
    def _parse_logical_or_expr(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

        ################################################################################################################

        left_node = Selection._parse_logical_and_expr(token_list)

        ################################################################################################################

        while token_list and token_list[0].type == 'LOGICAL_OR_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_logical_and_expr(token_list)

            left_node = Selection.BinaryOpNode(left_node, op, right_node)

        ################################################################################################################

        return left_node

    ####################################################################################################################

    @staticmethod
    def parse(expression: str) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

        """
        Evaluates the specified expression and returns the associated Abstract Syntax Tree (AST).

        .. code-block:: ebnf

            expression       = logical_or_expr ;

            logical_or_expr  = logical_and_expr { LOGICAL_OR_OP logical_and_expr } ;
            logical_and_expr = comparison_expr { LOGICAL_AND_OP comparison_expr } ;

            comparison_expr  = not_expr { COMPARISON_OP not_expr } ;

            not_expr         = [ NOT_OP ] term ;

            term             = "(" boolean_expr ")"
                             | FLOAT_NUM
                             | INT_NUM
                             | COL_NAME [(BITWISE_OR_OP | LOGICAL_AND_OP) INT_NUM]
                             ;

            NOT_OP           = "~" ;
            COMPARISON_OP    = "==" | "!=" | "<=" | ">=" | "<" | ">" ;
            LOGICAL_AND_OP   = "&&" | "and" ;
            LOGICAL_OR_OP    = "||" | "or" ;
            BITWISE_AND_OP   = "&" ;
            BITWISE_OR_OP    = "|" ;

        Parameters
        ----------
        expression : str
            The expression to be evaluated.

        Returns
        -------
        typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]
            The Abstract Syntax Tree (AST).
        """

        token_list = list(Selection._tokenize(expression))

        result = Selection._parse_logical_or_expr(token_list)

        if token_list:

            raise ValueError(f'Syntax error: unexpected token type `{token_list[0].type}`')

        return result

    ####################################################################################################################
    # SELECTION                                                                                                        #
    ####################################################################################################################

    @staticmethod
    def _evaluate(node: typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode], table: np.ndarray) -> typing.Union[np.ndarray, float]:

        ################################################################################################################
        # UNARY OP                                                                                                     #
        ################################################################################################################

        if isinstance(node, Selection.UnaryOpNode):

            right_value = Selection._evaluate(node.right, table)

            if node.op == '~':
                return np.logical_not(right_value)

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
            elif node.op == '&&' or node.op == 'and':
                return np.logical_and(left_value, right_value)
            elif node.op == '||' or node.op == 'or':
                return np.logical_or(left_value, right_value)
            elif node.op == '&':
                return np.bitwise_and(left_value, right_value)
            elif node.op == '|':
                return np.bitwise_or(left_value, right_value)

        ################################################################################################################
        # FLOAT NUM                                                                                                    #
        ################################################################################################################

        if isinstance(node, Selection.FloatNumNode):

            return float(node.value)

        ################################################################################################################
        # INT NUM                                                                                                      #
        ################################################################################################################

        if isinstance(node, Selection.IntNumNode):

            return int(node.value)

        ################################################################################################################
        # COL_NAME                                                                                                     #
        ################################################################################################################

        if isinstance(node, Selection.ColNameNode):

            return table[node.value]

        ################################################################################################################

        raise ValueError('Internal error, please contact the LE3 VMPZ-ID team')

    ####################################################################################################################

    @staticmethod
    def to_string(node: typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode], is_root: bool = True) -> str:

        """
        Converts the specified Abstract Syntax Tree (AST) into the associated expression.

        Parameters
        ----------
        node : typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]
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
        # FLOAT NUM                                                                                                    #
        ################################################################################################################

        elif isinstance(node, Selection.FloatNumNode):

            return str(float(node.value))

        ################################################################################################################
        # INT NUM                                                                                                      #
        ################################################################################################################

        elif isinstance(node, Selection.IntNumNode):

            return str(int(node.value))

        ################################################################################################################
        # COL_NAME                                                                                                     #
        ################################################################################################################

        elif isinstance(node, Selection.ColNameNode):

            return node.value

        ################################################################################################################

        raise ValueError('Internal error, please contact the LE3 VMPZ-ID team')

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

        ################################################################################################################

        expression = expression.strip()

        ################################################################################################################

        if not expression:

            return expression, np.full_like(table, 1, dtype = bool)

        else:

            ast = Selection.parse(expression)

            return (
                Selection.to_string(ast),
                Selection._evaluate(ast, table),
            )

########################################################################################################################
