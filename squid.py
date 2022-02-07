from __future__ import annotations
from typing import (Generator, TypeVar, Tuple, Generic, Optional, Callable,
                    Union, List, Any, Dict)

T = TypeVar('T')
S = TypeVar('S')
R = TypeVar('R')
Gen = Generator[T, None, None]

###############################################################################
# Misc
###############################################################################

indent = 0


def loud(f: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args: Any, **kwargs: Any) -> T:
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


def loud_generator(a: Gen[T]) -> Gen[T]:
    for e in a:
        print(e)
        yield e


def gen_of_file(filename: str) -> Gen[str]:
    with open(filename, 'r') as f:
        for line in f:
            yield from line


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
            raise ParseError("The pop was required, no value present")
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

STR_REP = {}
TRIV_LEXES = {}
KEYWORDS = {}


def define_triv_case(c: str) -> Tuple[int, str]:
    idx = next(lex_idx)
    TRIV_LEXES[c] = idx
    STR_REP[idx] = c
    return idx, c


def define_keyword(c: str) -> Tuple[int, str]:
    idx = next(lex_idx)
    KEYWORDS[c] = idx
    STR_REP[idx] = c
    return idx, c


def define_lex(message: str) -> int:
    idx = next(lex_idx)
    STR_REP[idx] = message
    return idx


LEX_LBRACK, CHR_LBRACK = define_triv_case('[')
LEX_RBRACK, CHR_RBRACK = define_triv_case(']')
LEX_LPARE, CHR_LPARE = define_triv_case('(')
LEX_RPARE, CHR_RPARE = define_triv_case(')')
LEX_LCURL, CHR_LCURL = define_triv_case('{')
LEX_RCURL, CHR_RCURL = define_triv_case('}')
LEX_COMMA, CHR_COMMA = define_triv_case(',')
LEX_SEMICOLON, CHR_SEMICOLON = define_triv_case(';')

LEX_OPERATOR = define_lex('operator')
CHR_OPERATOR = set((':', '+', '-', '*', '/', '%',
                    '<', '>', '=', '$', '|', '&', '!'))

LEX_LIT_INT = define_lex('int literal')
LEX_LIT_DOUBLE = define_lex('double literal')
LEX_LIT_STR = define_lex('str literal')
LEX_LIT_CHR = define_lex('character literal')
LEX_IDENTIFIER = define_lex('identifier')

LEX_PROD, CHR_PROD = define_keyword('=>')
LEX_FUN, CHR_FUN = define_keyword('fun')
LEX_FUN_DELIM, CHR_FUN_DELIM = define_keyword('->')
LEX_ASSIGN, CHR_ASSIGN = define_keyword(':=')
LEX_LCEOP_OSTART, CHR_LCEOP_OSTART = define_keyword('>>')
LEX_LCEOP_CSTART, CHR_LCEOP_CSTART = define_keyword('-<')
LEX_LCEOP_OEND, CHR_LCEOP_OEND = define_keyword('<<')
LEX_LCEOP_CEND, CHR_LCEOP_CEND = define_keyword('>-')

Lexem = Tuple[int, str]


def str_of_lexid(lexid: int) -> str:
    return STR_REP[lexid]


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


# ScopeStack
###############################################################################


K = TypeVar('K')
V = TypeVar('V')


class ScopeStack(Generic[K, V]):

    def __init__(self, stack: Optional[List[Dict[K, V]]] = None):
        self.stack: List[Dict[K, V]] = stack if stack is not None else []

    def lookup(self, symbol: K) -> Optional[V]:
        for dic in reversed(self.stack):
            if symbol in dic:
                return dic[symbol]
        return None

    def put_on(self, symbol: K, value: V, layer: int) -> None:
        self.stack[layer][symbol] = value

    def put(self, symbol: K, value: V, min_layer: int = 0) -> None:
        assert self.stack

        for index in range(len(self.stack) - 1, min_layer - 1, -1):
            if symbol in self.stack[index]:
                self.stack[index][symbol] = value
                return

        self.stack[-1][symbol] = value

    def put_on_last(self, symbol: K, value: V) -> None:
        assert self.stack
        self.stack[-1][symbol] = value

    def add_scope(self) -> None:
        self.stack.append({})

    def pop_scope(self) -> None:
        self.stack.pop()

    def copy(self, layers: Optional[int] = None) -> 'ScopeStack[K, V]':
        if layers is None:
            return ScopeStack(self.stack[:])
        return ScopeStack(self.stack[:layers])

    def __contains__(self, key: K) -> bool:
        return self.lookup(key) is not None


class Interpret:

    def __init__(self) -> None:
        self.sstack: ScopeStack[str, Value] = ScopeStack()


class InterError(BaseException):
    pass


###############################################################################
# Parser
###############################################################################


class ParseError(BaseException):
    pass


class PatternReject(BaseException):
    pass


class Grammar:

    def __init__(self) -> None:
        self.operator_table: List[Tuple[bool,
                                        List[str], List[str], List[str]]] = []


class Location:
    def __init__(self, filename: str, line_num: int):
        self.filename = filename
        self.line_num = line_num

    def __str__(self) -> str:
        return f'{self.filename}:{self.line_num}'


###############################################################################
# Values
###############################################################################


class SequenceObject:
    pass


class FunctionObject:

    def __init__(self, f: Callable[..., Value], argument_count: int):
        super().__init__()
        self.f = f
        self.argument_count = argument_count
        self.closure: Dict[str, Value] = {}

    def apply(self, inter: Interpret, *args: Value) -> Value:
        if len(args) != self.argument_count:
            raise InterError(f'function takes {self.argument_count} arguments'
                             + f', but {len(args)} were provided.')
        return self.f(inter, *args)


Value = Union[int, float, str, FunctionObject, SequenceObject]

###############################################################################
# Ast
###############################################################################


class AstElement(Generic[T]):

    def __init__(self, location: Optional[Location] = None):
        self.location = location

    def inter_error(self, message: str) -> InterError:
        return InterError(f'{str(self.location)} - {message}')

    def interpret(self, inter: Interpret) -> T:
        ...


class Constant(AstElement[Value]):

    def __init__(self, value: Value):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        if(isinstance(self.value, str)):
            return f"\"{self.value}\""
        return f"{self.value}"

    def interpret(self, _: Interpret) -> Value:
        return self.value


class Identifier(AstElement[Value]):

    def __init__(self,
                 name: str):
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return self.name

    def interpret(self, inter: Interpret) -> Value:
        val = inter.sstack.lookup(self.name)
        if val is None:
            raise self.inter_error(f"'{self.name}' not defined")
        return val


class FunctionApplication(AstElement[Value]):

    def __init__(self,
                 fun: Expression,
                 *arguments: Expression):
        super().__init__()
        self.fun = fun
        self.arguments = arguments

    def __str__(self) -> str:
        return (f"({str(self.fun)})" +
                ''.join(f" ({str(a)})" for a in self.arguments))

    def interpret(self, inter: Interpret) -> Value:
        fun_val = self.fun.interpret(inter)

        if not isinstance(fun_val, FunctionObject):
            raise self.inter_error('value is not a function')
        try:
            return fun_val.apply(inter,
                                 *[a.interpret(inter) for a in self.arguments])
        except InterError as e:
            raise self.inter_error(str(e))


class FunctionDefinition(AstElement[FunctionObject]):

    def __init__(self,
                 formal_arguments: List[str],
                 expr: Expression,
                 producing: Optional[str] = None):
        super().__init__()
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

    def interpret(self, _: Interpret) -> FunctionObject:
        def f(inter: Interpret, *args: Value) -> Value:
            inter.sstack.add_scope()
            for name, arg in zip(self.formal_arguments, args):
                inter.sstack.put_on_last(name, arg)
            try:
                res = self.expr.interpret(inter)
                return res
            except InterError:
                raise
            finally:
                inter.sstack.pop_scope()
        return FunctionObject(f, len(self.formal_arguments))


class SequenceDefinition(AstElement[SequenceObject]):

    def __init__(self, elements: List[Expression]):
        super().__init__()
        self.elements = elements

    def __str__(self) -> str:
        return (CHR_LBRACK
                + f'{CHR_COMMA} '.join(str(e) for e in self.elements)
                + CHR_RBRACK)


Atom = Union[FunctionApplication,
             Constant,
             FunctionDefinition,
             SequenceDefinition,
             Identifier]


Expression = Atom


class Assignment(AstElement[Tuple[str, Value]]):

    def __init__(self, name: str, expr: Expression):
        super().__init__()
        self.name = name
        self.expr = expr

    def __str__(self) -> str:
        return self.name + f' {CHR_ASSIGN} ' + str(self.expr)

    def interpret(self, inter: Interpret) -> Tuple[str, Value]:
        value = self.expr.interpret(inter)
        inter.sstack.put_on_last(self.name, value)
        return self.name, value


class Document(AstElement[Dict[str, Value]]):

    def __init__(self, *assignments: Assignment) -> None:
        super().__init__()
        self.assignments = assignments

    def interpret(self, inter: Interpret) -> Dict[str, Value]:
        res: Dict[str, Value] = {}
        for assignment in self.assignments:
            key, value = assignment.interpret(inter)
            res[key] = value
        return res


###############################################################################
# Parsing
###############################################################################


def match_token(state: ParsingState[Lexem, Any],
                *lex_id: int) -> Optional[Lexem]:
    return state.match_pred(lambda x: x[0] in lex_id)


def req_token(state: ParsingState[Lexem, Any], *lex_id: int) -> Lexem:
    expected = ', '.join(str_of_lexid(i) for i in lex_id)

    head = None
    try:
        head = state.rpop()
    except ParseError:
        head = (-1, 'eof')

    if head[0] not in lex_id:
        raise RuntimeError(f'expected {expected} but got {head[1]}')
    return head


def parse_document(state: ParsingState[Lexem, Grammar]) -> Document:
    res: List[Assignment] = []
    while state:
        res.append(parse_assignment(state))
    return Document(*res)


def parse_assignment(state: ParsingState[Lexem, Grammar]) -> Assignment:
    name = state.req_pred(is_lex(LEX_IDENTIFIER))
    state.req_pred(is_lex(LEX_ASSIGN))
    expr = parse_expression(state)
    req_token(state, LEX_SEMICOLON)
    return Assignment(name[1], expr)


def parse_application(state: ParsingState[Lexem, Grammar]) -> Atom:
    fun = parse_atom(state)
    arguments: List[Expression] = []
    while True:
        try:
            arguments.append(parse_atom(state))
        except PatternReject:
            break
    if arguments == []:
        return fun

    return FunctionApplication(fun, *arguments)


def parse_identifier(state: ParsingState[Lexem, Grammar]) -> Identifier:
    token = req_token(state, LEX_IDENTIFIER)
    return Identifier(token[1])


def parse_function_definition(state: ParsingState[Lexem, Grammar]) \
        -> FunctionDefinition:

    req_token(state, LEX_FUN)

    arguments: List[str] = []
    while (m := match_token(state, LEX_IDENTIFIER)) is not None:
        arguments.append(m[1])

    producing = None

    if match_token(state, LEX_PROD):
        identifier = parse_identifier(state)
        if identifier is not None:
            producing = identifier.name

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
        return parse_identifier(state)

    raise PatternReject(f"Invalid token: {state.peek()}")


def parse_expression_level_unary(state: ParsingState[Lexem, Grammar],
                                 level: int) -> Expression:

    _, _, prefix, posfix = state.data.operator_table[level - 1]
    prefix_tokens = [(LEX_OPERATOR, op) for op in prefix]
    posfix_tokens = [(LEX_OPERATOR, op) for op in posfix]

    prefix_stack = []

    while (m := state.match(*prefix_tokens)) is not None:
        prefix_stack.append(Identifier(m[1]))

    body = parse_expression_level(state, level - 1)

    for prefix_operator in reversed(prefix_stack):
        body = FunctionApplication(prefix_operator, body)

    while (m := state.match(*posfix_tokens)) is not None:
        body = FunctionApplication(Identifier(m[1]), body)

    return body


def parse_expression_level(state: ParsingState[Lexem, Grammar],
                           level: int) -> Expression:

    if level == 0:
        atom = parse_application(state)
        return atom

    associativity, infix, _, _ = state.data.operator_table[level - 1]

    elements = [parse_expression_level_unary(state, level)]
    operators = []
    infix_tokens = [(LEX_OPERATOR, op) for op in infix]

    while (m := state.match(*infix_tokens)) is not None:
        operators.append(Identifier(m[1]))
        elements.append(parse_expression_level_unary(state, level))

    if associativity:
        res = elements[-1]
        for i in range(len(elements) - 2, -1, -1):
            res = FunctionApplication(
                operators[i], elements[i], res)
    else:
        res = elements[0]
        for i in range(1, len(elements)):
            res = FunctionApplication(
                operators[i - 1], res, elements[i])

    return res


def parse_expression(state: ParsingState[Lexem, Grammar]) -> Expression:

    root_level = len(state.data.operator_table)

    operator_stack = []

    while match_token(state, LEX_LCEOP_CSTART) is not None:
        operator_stack.append(parse_expression(state))
        req_token(state, LEX_LCEOP_OEND)

    body = parse_expression_level(state, root_level)

    for operator in reversed(operator_stack):
        body = FunctionApplication(operator, body)

    middle_operator = None

    while match_token(state, LEX_LCEOP_OSTART) is not None:
        operator = parse_expression(state)
        token = req_token(state, LEX_LCEOP_OEND, LEX_LCEOP_CEND)
        if token[0] == LEX_LCEOP_OEND:
            middle_operator = operator
            break
        body = FunctionApplication(operator, body)

    if middle_operator:
        right = parse_expression(state)
        return FunctionApplication(middle_operator, body, right)
    return body


###############################################################################
# Builting building
###############################################################################


def define_builtin(inter: Interpret, name: str, val: Value) -> None:
    inter.sstack.put(name, val)


def builtin_function(inter: Interpret, name: str, argument_count: int) \
        -> Callable[[Callable[..., Value]], Callable[..., Value]]:
    def builtin_function_d(fun: Callable[..., Value]) -> Callable[..., Value]:
        define_builtin(inter, name, FunctionObject(fun, argument_count))
        return fun
    return builtin_function_d


def builtin_operator(inter: Interpret, name: str, arity: int, layer: int,
                     associativity: bool) \
        -> Callable[[Callable[..., Value]], Callable[..., Value]]:
    assert arity in {1, 2}

    def builtin_function_d(fun: Callable[..., Value]) -> Callable[..., Value]:
        define_builtin(inter, name, FunctionObject(fun, arity))

        table = grammar.operator_table[layer * 2 + (1 if associativity else 0)]

        if arity == 2:
            table[1].append(name)
        elif arity == 1 and not associativity:
            table[2].append(name)
        elif arity == 1 and associativity:
            table[3].append(name)

        return fun
    return builtin_function_d


###############################################################################
# Application
###############################################################################


state = ParsingState(gen_of_file('test.sq'), None)

grammar = Grammar()
OP_LAYER_COUNT = 5
for i in range(5):
    grammar.operator_table.append((False, [], [], []))
    grammar.operator_table.append((True, [], [], []))

lex_state = ParsingState(lexer(state), grammar)

inter = Interpret()
inter.sstack.add_scope()


@builtin_operator(inter, '+', 2, 0, False)
def op_plus(_: Interpret, a: Value, b: Value) -> int:
    assert(isinstance(a, int))
    assert(isinstance(b, int))
    return a + b


document = parse_document(lex_state)


print(document.interpret(inter))
