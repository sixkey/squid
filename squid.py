from __future__ import annotations
from typing import (Generator, TypeVar, Tuple, Generic, Optional, Callable,
                    Union, List, Any)

T = TypeVar('T')
S = TypeVar('S')
R = TypeVar('R')
Gen = Generator[T, None, None]

###############################################################################
# Misc
###############################################################################

indent = 0


def loud(f : Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args : Any, **kwargs : Any) -> T:
        global indent
        print(' ' * indent + f.__name__, *args, **kwargs)
        res = None
        try:
            indent += 4
            res = f(*args, **kwargs)
            return res
        finally:
            indent -= 4
            print(' ' * indent + str(res))

    return wrapper

###############################################################################
# ParsingState
###############################################################################


class ParsingError(BaseException):
    pass


class ParsingState(Generic[T, S]):

    def __init__(self, gen: Gen[T], data: S):
        self.gen = gen
        self.data = data
        self.value: Optional[T] = next(self.gen)

    def peek(self) -> Optional[T]:
        return self.value

    def rpeek(self) -> T:
        res = self.peek()
        if res is None:
            raise RuntimeError("The peek was required, no value present")
        return res

    def pop(self) -> Optional[T]:
        res = self.value
        try:
            self.value = next(self.gen)
        except StopIteration:
            self.value = None
        return res

    def rpop(self) -> T:
        res = self.pop()
        if res is None:
            raise RuntimeError("The pop was required, no value present")
        return res

    def required(self, value: T) -> T:
        head = self.rpop()
        if head != value:
            raise RuntimeError(f"{head} is not {value}")
        return head

    def req_pred(self, pred: Callable[[T], bool]) -> T:
        head = self.rpop()
        if not pred(head):
            raise RuntimeError(f"{head} failed predicate {pred}")
        return head

    def match(self, *values: T) -> Optional[T]:
        popped = self.peek()
        if popped in values:
            return self.pop()
        return None

    def match_pred(self, pred: Callable[[T], bool]) -> Optional[T]:
        popped = self.peek()
        if popped is not None and pred(popped):
            return self.pop()
        return None

    def __bool__(self) -> bool:
        return self.value is not None

###############################################################################
# Lexer
###############################################################################


def nats() -> Gen[int]:
    i = 0
    while True:
        yield i
        i += 1


lex_idx = nats()

LEX_LBRACK = next(lex_idx)
CHR_LBRACK = '['
LEX_RBRACK = next(lex_idx)
CHR_RBRACK = ']'
LEX_LPARE = next(lex_idx)
CHR_LPARE = '('
LEX_RPARE = next(lex_idx)
CHR_RPARE = ')'
LEX_LCURL = next(lex_idx)
CHR_LCURL = '{'
LEX_RCURL = next(lex_idx)
CHR_RCURL = '}'
LEX_COMMA = next(lex_idx)
CHR_COMMA = ','
LEX_SEMICOLON = next(lex_idx)
CHR_SEMICOLON = ';'


LEX_OPERATOR = next(lex_idx)
CHR_OPERATOR = set((':', '+', '-', '*', '/', '%', '<', '>', '=', '$'))

LEX_LIT_INT = next(lex_idx)
LEX_LIT_DOUBLE = next(lex_idx)
LEX_LIT_STR = next(lex_idx)
LEX_LIT_CHR = next(lex_idx)
LEX_IDENTIFIER = next(lex_idx)

TRIV_LEXES = {
    CHR_LBRACK: LEX_LBRACK,
    CHR_RBRACK: LEX_RBRACK,
    CHR_LPARE: LEX_LPARE,
    CHR_RPARE: LEX_RPARE,
    CHR_LCURL: LEX_LCURL,
    CHR_RCURL: LEX_RCURL,
    CHR_COMMA: LEX_COMMA,
    CHR_SEMICOLON: LEX_SEMICOLON
}

LEX_PROD = next(lex_idx)
CHR_PROD = '=>'
LEX_FUN = next(lex_idx)
CHR_FUN = 'fun'
LEX_FUN_DELIM = next(lex_idx)
CHR_FUN_DELIM = '->'
LEX_ASSIGN = next(lex_idx)
CHR_ASSIGN = ':='

KEYWORDS = {
    CHR_PROD: LEX_PROD,
    CHR_FUN: LEX_FUN,
    CHR_ASSIGN: LEX_ASSIGN,
    CHR_FUN_DELIM: LEX_FUN_DELIM
}

Lexem = Tuple[int, str]


def lexer(state: ParsingState[str, None]) -> Gen[Lexem]:

    while state:
        character = state.rpeek()
        if character == '"':
            yield from lexer_string(state)
        elif character.isdigit():
            yield from lex_number(state)
        elif character.isalpha():
            yield from lex_identifier(state)
        elif character in CHR_OPERATOR:
            yield from lex_operator(state)
        elif character in TRIV_LEXES:
            state.rpop()
            yield TRIV_LEXES[character], character
        elif character.isspace():
            state.rpop()
        else:
            raise ParsingError(f'Unexpected character: {character}')


def lexer_string(state: ParsingState[str, None]) -> Gen[Lexem]:
    state.required('"')
    buffer = ''
    escape = False
    while state and (escape or state.rpeek() != '"'):
        character = state.rpop()
        if escape:
            if character == 'n':
                buffer += '\n'
            if character == '"':
                buffer += '"'
            else:
                raise ParsingError(f"Unknown escape sequence: \\{character}")
        elif character == '\\':
            escape = True
        else:
            buffer += character
    state.required('"')
    yield LEX_LIT_STR, buffer


def lex_number(state: ParsingState[str, None]) -> Gen[Lexem]:
    buffer = ''
    lex_id = LEX_LIT_INT
    while state and state.rpeek().isdigit():
        buffer += state.rpop()
    if state and state.rpeek() == '.':
        buffer += state.rpop()
        while state and state.rpeek().isdigit():
            buffer += state.rpop()
        LEX_LIT_DOUBLE
    yield lex_id, buffer


def lex_operator(state: ParsingState[str, None]) -> Gen[Lexem]:
    buffer = ''
    while state and state.rpeek() in CHR_OPERATOR:
        buffer += state.rpop()

    if buffer in KEYWORDS:
        yield KEYWORDS[buffer], buffer
    else:
        yield LEX_OPERATOR, buffer


def lex_identifier(state: ParsingState[str, None]) -> Gen[Lexem]:
    buffer = state.req_pred(lambda x: x.isalpha())
    while state and state.rpeek().isalnum() or state.rpeek() == '_':
        buffer += state.rpop()
    if buffer in KEYWORDS:
        yield KEYWORDS[buffer], buffer
    else:
        yield LEX_IDENTIFIER, buffer


def is_lex(lex_id: int) -> Callable[[Lexem], bool]:
    def is_lex_w(lexem: Lexem) -> bool:
        return lexem[0] == lex_id
    return is_lex_w


###############################################################################
# Parser
###############################################################################


class PatternReject(BaseException):
    pass


class Grammar:

    def __init__(self) -> None:
        self.operator_table: List[Tuple[bool, List[str]]] = []


class AstElement:
    pass


Value = Union[int, float, str]


class Constant(AstElement):

    def __init__(self, value: Value):
        self.value = value

    def __str__(self) -> str:
        if(isinstance(self.value, str)):
            return f"\"{self.value}\""
        return f"{self.value}"


class FunctionCall(AstElement):

    def __init__(self,
                 name: str,
                 *arguments: Expression,
                 is_operator: bool = False):
        self.name = name
        self.arguments = arguments
        self.is_operator = is_operator

    def __str__(self) -> str:
        if (self.is_operator):
            assert(len(self.arguments) == 2)
            return f"({self.arguments[0]} {self.name} {self.arguments[1]})"
        return f"{self.name}" + ''.join(f" ({str(a)})" for a in self.arguments)


class FunctionDefinition(AstElement):

    def __init__(self,
                 formal_arguments: List[str],
                 expr: Expression,
                 producing: Optional[str] = None):
        self.formal_arguments = formal_arguments
        self.expr = expr
        self.producing = producing

    def __str__(self) -> str:
        return (CHR_FUN
                + "".join(' ' + s for s in self.formal_arguments)
                + (f' {CHR_PROD} {self.producing}'
                    if self.producing is not None
                    else '')
                + f' {CHR_FUN_DELIM} '
                + str(self.expr))


class SequenceDefinition(AstElement):

    def __init__(self, elements: List[Expression]):
        self.elements = elements

    def __str__(self) -> str:
        return (CHR_LBRACK
                + CHR_COMMA.join(str(e) for e in self.elements)
                + CHR_RBRACK)


Atom = Union[FunctionCall, Constant, FunctionDefinition, SequenceDefinition]

Expression = Atom


class Assignment(AstElement):

    def __init__(self, name: str, expr: Expression):
        self.name = name
        self.expr = expr

    def __str__(self) -> str:
        return self.name + f' {CHR_ASSIGN} ' + str(self.expr)


def parse_document(state: ParsingState[Lexem, Grammar]) -> List[Assignment]:
    res = []
    while state:
        res.append(parse_assignment(state))
    return res


def parse_assignment(state: ParsingState[Lexem, Grammar]) -> Assignment:
    name = state.req_pred(is_lex(LEX_IDENTIFIER))
    state.req_pred(is_lex(LEX_ASSIGN))
    expr = parse_expression(state)
    req_token(state, LEX_SEMICOLON)
    return Assignment(name[1], expr)


def match_token(state: ParsingState[Lexem, Any],
                lex_id: int) -> Optional[Lexem]:
    return state.match_pred(lambda x: x[0] == lex_id)


def req_token(state: ParsingState[Lexem, Any], lex_id: int) -> Lexem:
    return state.req_pred(lambda x: x[0] == lex_id)


def parse_application(state: ParsingState[Lexem, Grammar]) -> FunctionCall:
    name = req_token(state, LEX_IDENTIFIER)
    arguments: List[Expression] = []
    while True:
        try:
            if (m := match_token(state, LEX_IDENTIFIER)) is not None:
                arguments.append(FunctionCall(m[1]))
            else:
                arguments.append(parse_atom(state))
        except PatternReject:
            break
    return FunctionCall(name[1], *arguments)


def parse_identifier(state: ParsingState[Lexem, Grammar]) -> str:
    token = req_token(state, LEX_IDENTIFIER)
    return token[1]


def parse_function_definition(state: ParsingState[Lexem, Grammar]) -> FunctionDefinition:

    req_token(state, LEX_FUN)

    arguments: List[str] = []
    while (m := match_token(state, LEX_IDENTIFIER)) is not None:
        arguments.append(m[1])

    producing = None

    if match_token(state, LEX_PROD):
        producing = parse_identifier(state)

    req_token(state, LEX_FUN_DELIM)

    expr = parse_expression(state)
    return FunctionDefinition(arguments, expr, producing=producing)


def parse_sequence(state: ParsingState[Lexem, Grammar]) -> Expression:

    req_token(state, LEX_LBRACK)

    elements = []

    while match_token(state, LEX_RBRACK) is None:
        elements.append(parse_expression(state))
        if (match_token(state, LEX_COMMA) is None):
            break

    req_token(state, LEX_RBRACK)

    return SequenceDefinition(elements)


def parse_atom(state: ParsingState[Lexem, Grammar]) -> Expression:

    token = state.peek()
    if (token is None):
        raise PatternReject(f"Invalid token: {state.peek()}")

    if (token[0] == LEX_LPARE):
        state.rpop()
        expr = parse_expression(state)
        state.required((LEX_RPARE, ')'))
        return expr

    if (token[0] == LEX_LIT_STR):
        state.rpop()
        return Constant(str(token[1]))
    if (token[0] == LEX_LIT_DOUBLE):
        state.rpop()
        return Constant(float(token[1]))
    if (token[0] == LEX_LIT_INT):
        state.rpop()
        return Constant(int(token[1]))
    if (token[0] == LEX_FUN):
        return parse_function_definition(state)
    if (token[0] == LEX_LBRACK):
        return parse_sequence(state)
    if (token[0] == LEX_IDENTIFIER):
        return parse_application(state)

    raise PatternReject(f"Invalid token: {state.peek()}")


def parse_expression_level(state: ParsingState[Lexem, Grammar],
                           level: int) -> Expression:

    if level == 0:
        atom = parse_atom(state)
        return atom

    associativity, operators_table = state.data.operator_table[level - 1]

    elements = [parse_expression_level(state, level - 1)]
    operators = []

    operator_tokens = [(LEX_OPERATOR, op) for op in operators_table]

    while (m := state.match(*operator_tokens)) is not None:
        operators.append(m[1])
        elements.append(parse_expression_level(state, level - 1))

    if associativity:
        res = elements[-1]
        for i in range(len(elements) - 2, -1, -1):
            res = FunctionCall(
                operators[i], elements[i], res, is_operator=True)
    else:
        res = elements[0]
        for i in range(1, len(elements)):
            res = FunctionCall(
                operators[i - 1], res, elements[i], is_operator=True)

    return res


def parse_expression(state: ParsingState[Lexem, Grammar]) -> Expression:
    return parse_expression_level(state, len(state.data.operator_table))


###############################################################################
# Application
###############################################################################

def gen_of_file(filename: str) -> Gen[str]:
    with open(filename, 'r') as f:
        for line in f:
            for c in line:
                yield c


state = ParsingState(gen_of_file('test.sq'), None)

grammar = Grammar()

grammar.operator_table.append((False, ['*', '/']))
grammar.operator_table.append((False, ['+', '-']))

lex_state = ParsingState(lexer(state), grammar)

print('\n'.join(str(s) for s in parse_document(lex_state)))
