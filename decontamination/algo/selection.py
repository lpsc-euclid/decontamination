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
import healpy as hp

########################################################################################################################

class Selection(object):

    """
    Data selection from expression string.
    """

    ####################################################################################################################
    # TOKENIZER                                                                                                        #
    ####################################################################################################################

    class Token(object):

        """
        A token.

        :private:
        """

        ################################################################################################################

        def __init__(self, _type: str, _value: str):

            self.type: str = _type
            self.value: str = _value

        ################################################################################################################

        def __str__(self):

            return f'<{self.type},`{self.value}`>'

    ####################################################################################################################

    _TOKEN_PATTERN = re.compile(
        r'([()])'
        r'|'
        r'(isfinite)'
        r'|'
        r'(~)'
        r'|'
        r'(==|!=|<=|>=|<|>)'
        r'|'
        r'(&&|and)|(\|\||or)'
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
            r'0o[0-7]+'
            r'|'
            r'0b[0-1]+'
            r'|'
            r'\d+'
        r'))'
        r'|'
        r'(\w+)'
        r'|'
        r'(\s+)'
        r'|'
        r'(.)'
    )

    ####################################################################################################################

    @staticmethod
    def _tokenize(expression: str) -> typing.Generator[Token, typing.Any, typing.Any]:

        for grouping, isfinite_op, not_op, comparison_op, logical_and_op, logical_or_op, bitwise_and_op, bitwise_or_op, float_num, int_num, col_name, blank, unknown in Selection._TOKEN_PATTERN.findall(expression):

            if grouping:
                yield Selection.Token('GROUPING', grouping)
            elif isfinite_op:
                yield Selection.Token('ISFINITE_OP', isfinite_op)
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
                pass  # IGNORE BLANK CHARACTERS #
            else:
                yield Selection.Token('UNKNOWN', unknown)

    ####################################################################################################################
    # PARSER                                                                                                           #
    ####################################################################################################################

    class UnaryOpNode(object):

        """
        An unary operator node.
        :private:
        """

        def __init__(self, _op: str, _right):

            self.op: str = _op
            self.right = _right

    ####################################################################################################################

    class BinaryOpNode(object):

        """
        A binary operator node.
        :private:
        """

        def __init__(self, _left, _op: str, _right):

            self.left = _left
            self.op: str = _op
            self.right = _right

    ####################################################################################################################

    class FloatNumNode(object):

        """
        A floating number node.
        :private:
        """

        def __init__(self, _value: str):

            self.value = float(_value)

    ####################################################################################################################

    class IntNumNode(object):

        """
        An integer number node.
        :private:
        """

        def __init__(self, _value: str):

            if _value.startswith('0x'):
                self.value = int(_value[2:], base = 16)
            elif _value.startswith('0o'):
                self.value = int(_value[2:], base = 8)
            elif _value.startswith('0b'):
                self.value = int(_value[2:], base = 2)
            else:
                self.value = int(_value)

    ####################################################################################################################

    class ColNameNode(object):

        """
        A column name node.
        :private:
        """

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

            if token_list and token_list[0].type in ['BITWISE_AND_OP', 'BITWISE_OR_OP']:

                op = token_list.pop(0).value

                token = Selection._pick_token(token_list, expected_type = 'INT_NUM')

                right_node = Selection.IntNumNode(token.value)

                left_node = Selection.BinaryOpNode(left_node, op, right_node)

            ############################################################################################################

            return left_node

        ################################################################################################################

        raise ValueError(f'Syntax error: unexpected token type `{token.type}`')

    ####################################################################################################################

    @staticmethod
    def _parse_isfinite_op(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

        if token_list and token_list[0].type == 'ISFINITE_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_term(token_list)

            return Selection.UnaryOpNode(op, right_node)

        return Selection._parse_term(token_list)

    ####################################################################################################################

    @staticmethod
    def _parse_not_op(token_list: typing.List[Token]) -> typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode]:

        if token_list and token_list[0].type == 'NOT_OP':

            op = token_list.pop(0).value

            right_node = Selection._parse_isfinite_op(token_list)

            return Selection.UnaryOpNode(op, right_node)

        return Selection._parse_isfinite_op(token_list)

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

            not_expr         = [ NOT_OP ] isfinite_expr ;

            isfinite_expr    = [ ISFINITE_OP ] term ;

            term             = "(" logical_or_expr ")"
                             | FLOAT_NUM
                             | INT_NUM
                             | COL_NAME [(BITWISE_OR_OP | BITWISE_AND_OP) INT_NUM]
                             ;

            ISFINITE_OP      = "isfinite" ;
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

            raise ValueError(f'Syntax error: unexpected token(s): {" ".join([token.__str__() for token in token_list])}')

        return result

    ####################################################################################################################
    # SELECTION                                                                                                        #
    ####################################################################################################################

    class MaskedColumn(object):

        """
        A masked column.
        :private:
        """

        def __init__(self, shape):

            self.shape: typing.Union[int, typing.Tuple[int]] = shape

    ####################################################################################################################

    @staticmethod
    def _isfinite(x: typing.Union[np.ndarray, np.ma.MaskedArray]) -> np.ndarray:

        y = np.asarray(x)

        if not hasattr(x, 'data')\
           or                    \
           not hasattr(x, 'mask'):

            return np.isfinite(y) & (y != hp.UNSEEN)

        else:

            return np.isfinite(y) & (y != hp.UNSEEN) & ~x.mask

    ####################################################################################################################

    @staticmethod
    def _evaluate(node: typing.Union[UnaryOpNode, BinaryOpNode, FloatNumNode, IntNumNode, ColNameNode], table: typing.Union[np.ndarray, np.ma.MaskedArray], ignore_masked_columns: bool = False) -> typing.Union['MaskedColumn', np.ndarray, float]:

        ################################################################################################################
        # UNARY OP                                                                                                     #
        ################################################################################################################

        if isinstance(node, Selection.UnaryOpNode):

            right_value = Selection._evaluate(node.right, table, ignore_masked_columns)

            if node.op == 'isfinite':
                return Selection._isfinite(right_value)
            if node.op == '~':
                return np.logical_not(right_value)

        ################################################################################################################
        # BINARY OP                                                                                                    #
        ################################################################################################################

        if isinstance(node, Selection.BinaryOpNode):

            left_value = Selection._evaluate(node.left, table, ignore_masked_columns)
            right_value = Selection._evaluate(node.right, table, ignore_masked_columns)

            if ignore_masked_columns:

                if isinstance(left_value, Selection.MaskedColumn):
                    return np.full_like(left_value.shape, True, dtype = bool) if node.op not in ['&', '|'] else left_value
                if isinstance(right_value, Selection.MaskedColumn):
                    return np.full_like(right_value.shape, True, dtype = bool) if node.op not in ['&', '|'] else right_value

            if node.op == '==':
                return left_value == right_value
            if node.op == '!=':
                return left_value != right_value
            if node.op == '<=':
                return left_value <= right_value
            if node.op == '>=':
                return left_value >= right_value
            if node.op == '<':
                return left_value < right_value
            if node.op == '>':
                return left_value > right_value
            if node.op == '&&' or node.op == 'and':
                return np.logical_and(left_value, right_value)
            if node.op == '||' or node.op == 'or':
                return np.logical_or(left_value, right_value)
            if node.op == '&':
                return np.bitwise_and(left_value, right_value)
            if node.op == '|':
                return np.bitwise_or(left_value, right_value)

        ################################################################################################################
        # FLOAT / INT NUM                                                                                              #
        ################################################################################################################

        if isinstance(node, Selection.FloatNumNode) or isinstance(node, Selection.IntNumNode):

            return node.value

        ################################################################################################################
        # COL_NAME                                                                                                     #
        ################################################################################################################

        if isinstance(node, Selection.ColNameNode):

            if ignore_masked_columns and np.all(Selection._isfinite(table[node.value])):

                return Selection.MaskedColumn(table[node.value].shape)

            else:

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

            if node.op[-1].isalnum():
                expr = f'{node.op} {right_expr}'
            else:
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
        # FLOAT / INT NUM                                                                                              #
        ################################################################################################################

        elif isinstance(node, Selection.FloatNumNode) or isinstance(node, Selection.IntNumNode):

            return str(node.value)

        ################################################################################################################
        # COL_NAME                                                                                                     #
        ################################################################################################################

        elif isinstance(node, Selection.ColNameNode):

            return node.value

        ################################################################################################################

        raise ValueError('Internal error, please contact the LE3 VMPZ-ID team')

    ####################################################################################################################

    @staticmethod
    def filter_table(expression: str, table: typing.Union[np.ndarray, np.ma.MaskedArray], ignore_masked_columns: bool = False) -> typing.Tuple[str, np.ndarray]:

        """
        Evaluates the specified expression and filters the table.

        Parameters
        ----------
        expression : str
            The expression to be evaluated.
        table : typing.Union[np.ndarray, np.ma.MaskedArray]
            The table to be filtered by the expression.
        ignore_masked_columns : bool, default: False
            Indicates whether to ignore the selection on masked columns.

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

            return expression, np.full_like(table, True, dtype = bool)

        else:

            ast = Selection.parse(expression)

            return Selection.to_string(ast), Selection._evaluate(ast, table, ignore_masked_columns)

########################################################################################################################
