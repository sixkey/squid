from __future__ import annotations
from typing import (Generator, TypeVar, Tuple, Generic, Optional, Callable,
                    Union, List, Any, Dict, Set, Iterable)
from ppretty import ppretty

###############################################################################
# TODO
###############################################################################
"""

Syntax:
    - let in

Operators:
    - operator definition
    - unary versus binary - different namespace (different function
                            definitions)

Evaluation:
    - interpreter error printing

Types:
    - type checking

Sequence:
    - sequence interpreting
    - intensional sequences
    - sets

Field:
    - basic definition

"""
###############################################################################
###############################################################################
###############################################################################


T = TypeVar('T')
S = TypeVar('S')
R = TypeVar('R')
K = TypeVar('K')
V = TypeVar('V')
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
# Lexing
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
    STR_REP[idx] = f"'{c}'"
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
                    '<', '>', '=', '$', '|', '&', '!', '.'))

LEX_LIT_INT = define_lex('int literal')
LEX_LIT_DOUBLE = define_lex('double literal')
LEX_LIT_STR = define_lex('str literal')
LEX_LIT_CHR = define_lex('character literal')
LEX_IDENTIFIER = define_lex('identifier')

LEX_PROD, CHR_PROD = define_keyword('=>')
LEX_FUN, CHR_FUN = define_keyword('fun')

LEX_OPBINL, CHR_OPBINL = define_keyword('opbinl')
LEX_OPBINR, CHR_OPBINR = define_keyword('opbinr')

LEX_FUN_DELIM, CHR_FUN_DELIM = define_keyword('->')
LEX_ASSIGN, CHR_ASSIGN = define_keyword(':=')
LEX_LCEOP_OSTART, CHR_LCEOP_OSTART = define_keyword('>>')
LEX_LCEOP_CSTART, CHR_LCEOP_CSTART = define_keyword('-<')
LEX_LCEOP_OEND, CHR_LCEOP_OEND = define_keyword('<<')
LEX_LCEOP_CEND, CHR_LCEOP_CEND = define_keyword('>-')

LEX_IF, CHR_IF = define_keyword('if')
LEX_THEN, CHR_THEN = define_keyword('then')
LEX_ELSE, CHR_ELSE = define_keyword('else')

LEX_EOF = define_lex('eof')
CHR_EOF = 'eof';

LexemCore = Tuple[int, str]
Lexem = Tuple[int, str, Tuple[int, int]]

def str_of_lexid(lexid: int) -> str:
    return STR_REP[lexid]


def lexer(state: ParsingState[str, None]) -> Gen[Lexem]:

    while state:
        character = state.rpeek()
        row_idx = state.row_counter
        col_idx = state.col_counter
        lexem = None
        if character == '"':
            lexem = lexer_string(state)
        elif character.isdigit():
            lexem = lex_number(state)
        elif character.isalpha():
            lexem = lex_identifier(state)
        elif character in CHR_OPERATOR:
            lexem = lex_operator(state)
        elif character in TRIV_LEXES:
            state.rpop()
            lexem = TRIV_LEXES[character], character
        elif character.isspace():
            state.rpop()
        else:
            raise ParsingError(f'Unexpected character: {character}')

        if lexem != None:
            yield lexem[0], lexem[1], (row_idx, col_idx)
    yield LEX_EOF, CHR_EOF, (state.row_counter, state.col_counter)


def lexer_string(state: ParsingState[str, None]) -> LexemCore:
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
    return LEX_LIT_STR, buffer


def lex_number(state: ParsingState[str, None]) -> LexemCore:
    buffer = ''
    lex_id = LEX_LIT_INT
    while state and state.rpeek().isdigit():
        buffer += state.rpop()
    if state and state.rpeek() == '.':
        buffer += state.rpop()
        while state and state.rpeek().isdigit():
            buffer += state.rpop()
        LEX_LIT_DOUBLE
    return lex_id, buffer


def lex_operator(state: ParsingState[str, None]) -> LexemCore:
    buffer = ''
    while state and state.rpeek() in CHR_OPERATOR:
        buffer += state.rpop()

    if buffer in KEYWORDS:
        return KEYWORDS[buffer], buffer
    else:
        return LEX_OPERATOR, buffer


def lex_identifier(state: ParsingState[str, None]) -> LexemCore:
    buffer = state.req_pred(lambda x: x.isalpha())
    while state and state.rpeek().isalnum() or state.rpeek() == '_':
        buffer += state.rpop()
    if buffer in KEYWORDS:
        return KEYWORDS[buffer], buffer
    else:
        return LEX_IDENTIFIER, buffer


###############################################################################
# Parsing
###############################################################################

class PatternReject(BaseException):
    pass

class ParsingError(BaseException):
    pass

class ParseError(BaseException):
    pass

class ParsingState(Generic[T, S]):

    def __init__(self, gen: Gen[T], data: S, row_element: Optional[T] = None):
        self.gen = gen
        self.data = data
        self.value: Optional[T] = next(self.gen)

        self.row_element : Optional[T] = row_element

        self.row_counter = 0
        self.col_counter = 0

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
            self.col_counter += 1
            if self.row_element is not None and self.value == self.row_element:
                self.row_counter += 1
                self.col_counter = 0
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
# Location
###############################################################################

Location = Tuple[int, int]

def str_of_location(location: Tuple[int, int]) -> str:
    return f'{location[0] + 1}:{location[1] + 1}'

###############################################################################
# ScopeStack
###############################################################################

class FutureBinding(Generic[T]):

    def __init__(self) -> None:
        self.value : Optional[T]


Binding = Union[T, FutureBinding[T]]

class ScopeStack(Generic[K, V]):

    def __init__(self, stack: Optional[List[Dict[K, Binding[V]]]] = None):
        self.stack: List[Dict[K, Binding[V]]] = (
                stack if stack is not None else [])

    def lookup(self, symbol: K) -> Optional[V]:
        value = self.lookup_future(symbol)
        if isinstance(value, FutureBinding):
            return value.value
        return value

    def lookup_future(self, symbol: K) -> Optional[Union[V, FutureBinding[V]]]:
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

    def put_on_last_future(self, symbol: K, value: Binding[V]) -> None:
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

    def __str__(self) -> str:
        return '\n'.join(str(l) for l in self.stack)


###############################################################################
# Interpret state
###############################################################################


class Interpret:

    def __init__(self) -> None:
        self.sstack: ScopeStack[str, Value] = ScopeStack()


class InterError(BaseException):
    pass


###############################################################################
# Values
###############################################################################

class Dependency(Generic[T]):

    def __init__(self, stage: int, children: List[AstElement[Any]], data: T,
                 give_amount: Optional[int] = None):
        self.stage = stage
        self.children = children
        self.data = data
        self.give_amount = give_amount if give_amount != None else len(children)

class CurrentStage(Generic[T]):

    def __init__(self, stage: int, args: List[Value], data: T):
        self.stage = stage
        self.args = args
        self.data = data

Response = Union[T, Dependency[R]]

class SequenceObject:
    pass

F = Callable[[Interpret, CurrentStage], Response[T, None]]

class FunctionObject:

    def __init__(self, f: F[Value],
                 argument_count: int, *partial: Value):
        super().__init__()
        self.f = f
        self.argument_count = argument_count
        self.partial = list(partial)

    def copy(self, *partial: Value) -> FunctionObject:
        return FunctionObject(self.f, self.argument_count - len(partial), *partial)

    def apply(self, inter: Interpret, stage: CurrentStage[bool]) -> Response[Value, None]:
        assert len(stage.args) <= self.argument_count
        if stage.data:
            return self.f(inter, CurrentStage(stage.stage, stage.args, None))
        if len(stage.args) == self.argument_count:
            return self.f(inter, CurrentStage(stage.stage, self.partial + stage.args, None))
        return self.copy(*stage.args)

    def __repr__(self) -> str:
        return f'FunctionObject {self.argument_count}'


Value = Union[bool, int, float, str, FunctionObject, SequenceObject]

###############################################################################
# Evaluation
###############################################################################

def take(a: List[T], count: int) -> List[T]:
    res = []
    for _ in range(count):
        res.append(a.pop())
    return res


def ast_interpret(inter: Interpret, root: AstElement[Any]) -> Any:

    pile : List[Value] = []
    stack : List[Tuple[int, int, Any, AstElement[Any]]] = [(0, 0, None, root)]

    while stack:

        stage, arg_count, data, element = stack.pop()
        stage = CurrentStage(stage, take(pile, arg_count), data)
        response = element.inter_step(inter, stage)

        if isinstance(response, Dependency):
            stack.append((response.stage, response.give_amount, response.data, element))
            for child in response.children:
                stack.append((0, 0, None, child))
        else:
            pile.append(response)

    assert len(pile) == 1
    return pile[0]


###############################################################################
# Grammar
###############################################################################

class Grammar:

    def __init__(self, operator_levels: int) -> None:
        self.operator_table: List[Tuple[bool,
                                        List[str], List[str], List[str]]] = []

        for _ in range(operator_levels):
            self.operator_table.append((False, [], [], []))
            self.operator_table.append((True, [], [], []))

        self.operators: Set[str] = set()
        self.operator_levels = operator_levels

    def add_operator(self, name: str, level: int, arity: int,
                     associativity: int):

        if level >= len(self.operator_table):
            raise RuntimeError(
                f'operator level {level} exceeded {len(self.operator_table)}')

        table = self.operator_table[level * 2 + (1 if associativity else 0)]

        if arity == 2:
            table[1].append(name)
        elif arity == 1 and not associativity:
            table[2].append(name)
        elif arity == 1 and associativity:
            table[3].append(name)

        self.operators.add(name)

###############################################################################
# Ast
###############################################################################

class AstElement(Generic[T]):

    def __init__(self, location: Location):
        self.location = location

    def inter_error(self, message: str) -> InterError:
        return InterError(f'{str_of_location(self.location)} - {message}')

    def interpret(self, inter: Interpret) -> T:
        ...

    def get_free_names(self) -> Set[str]:
        ...

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
            -> Response[Value]:
        ...


class Constant(AstElement[Value]):

    def __init__(self, location: Location, value: Value):
        super().__init__(location)
        self.value = value

    def __str__(self) -> str:
        if(isinstance(self.value, str)):
            return f"\"{self.value}\""
        return f"{self.value}"

    def interpret(self, _: Interpret) -> Value:
        return self.value

    def get_free_names(self) -> Set[str]:
        return set()

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
            -> Response[Value]:
        return self.value


class Identifier(AstElement[Value]):

    def __init__(self,
                 location: Location,
                 name: str):
        super().__init__(location)
        self.name = name

    def __str__(self) -> str:
        return self.name

    def interpret(self, inter: Interpret) -> Value:
        val = inter.sstack.lookup(self.name)
        if val is None:
            raise self.inter_error(f"'{self.name}' not defined")
        return val

    def get_free_names(self) -> Set[str]:
        return set([self.name])

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
            -> Response[Value]:
        val = inter.sstack.lookup(self.name)
        if val is None:
            raise self.inter_error(f"'{self.name}' not defined")
        return val


class FunctionApplication(AstElement[Value]):

    def __init__(self,
                 fun: Expression,
                 *arguments: Expression):
        super().__init__(fun.location)
        self.fun = fun
        self.arguments = list(arguments)

    def __str__(self) -> str:
        return (f"({str(self.fun)})" +
                ''.join(f" ({str(a)})" for a in self.arguments))

    def get_free_names(self) -> Set[str]:
        res = self.fun.get_free_names()
        for arg in self.arguments:
            res.update(arg.get_free_names())
        return res

    def inter_step(self, inter: Interpret, stage: CurrentStage[Any]) \
            -> Response[Value, Any]:

        if stage.stage == 0:
            return Dependency(1, [self.fun], None)

        if stage.stage == 1:
            fun_val = stage.args[0]

            # TODO: this should be type error
            if not isinstance(fun_val, FunctionObject):
                raise self.inter_error(f'value {fun_val} is not a function')

            argument_count = min(len(self.arguments), fun_val.argument_count)
            return Dependency(2, self.arguments, (0, fun_val, len(self.arguments), False), argument_count)

        if stage.stage == 2:

            assert (stage.data is not None)

            fun_stage, fun_val, reminder, force = stage.data
            response = fun_val.apply(inter, CurrentStage(fun_stage, stage.args, force))

            if isinstance(response, Dependency):
                return Dependency(
                    2, response.children, (response.stage, fun_val, reminder, True))

            if isinstance(response, FunctionObject) and reminder == 0:
                argument_delta = min(reminder, response.argument_count)
                return Dependency(2, [], (0, response, argument_delta, False), argument_delta)

            reminder = max(reminder - fun_val.argument_count, 0)

            if reminder != 0:
                raise self.inter_error(
                    f"Function takes {len(self.arguments) - reminder} " +
                    f"arguments, but {len(self.arguments)} were given.")

            return response

        assert False


def list_of_opt(opt: Optional[T]) -> List[T]:
    return [] if opt is None else [opt]


class FunctionDefinition(AstElement[FunctionObject]):

    def __init__(self,
                 location: Location,
                 formal_arguments: List[str],
                 expr: Expression,
                 producing: Optional[str] = None):
        super().__init__(location)
        self.formal_arguments = formal_arguments
        self.expr = expr
        self.producing = producing
        self.capture_names = self.get_free_names()

    def __str__(self) -> str:
        return (CHR_FUN
                + "".join(' ' + s for s in self.formal_arguments)
                + (f' {CHR_PROD} {self.producing}'
                    if self.producing is not None
                    else '')
                + f' {CHR_FUN_DELIM} '
                + str(self.expr))

    def inter_step(self, inter: Interpret, stage: CurrentStage) -> Response[FunctionObject, None]:

        closure : Dict[str, Binding[Value]] = {}

        for name in self.capture_names:
            value = inter.sstack.lookup_future(name)
            if value is None:
                raise self.inter_error(f"value '{name}' not defined")
            closure[name] = value

        def f(inter: Interpret, stage: CurrentStage[None]) -> Response[Value, None]:

            if stage.stage == 0:

                inter.sstack.add_scope()
                for name, value in closure.items():
                    inter.sstack.put_on_last_future(name, value)

                inter.sstack.add_scope()
                for name, arg in zip(self.formal_arguments, stage.args):
                    inter.sstack.put_on_last(name, arg)

                return Dependency(1, [self.expr], None)

            if stage.stage == 1:

                inter.sstack.pop_scope()
                inter.sstack.pop_scope()

                return stage.args[0]

            assert False

        return FunctionObject(f, len(self.formal_arguments))

    def get_free_names(self) -> Set[str]:
        names = self.expr.get_free_names()
        remove = set(self.formal_arguments + list_of_opt(self.producing))
        names = names.difference(remove)
        return names


class SequenceDefinition(AstElement[SequenceObject]):

    def __init__(self, location: Location, elements: List[Expression]):
        super().__init__(location)
        self.elements = elements

    def __str__(self) -> str:
        return (CHR_LBRACK
                + f'{CHR_COMMA} '.join(str(e) for e in self.elements)
                + CHR_RBRACK)


class IfStmt(AstElement[Value]):

    def __init__(self, location: Location,
                 cond: Expression, branch_true: Expression,
                 branch_false: Expression) -> None:
        super().__init__(location)
        self.cond = cond
        self.branch_true = branch_true
        self.branch_false = branch_false

    def interpret(self, inter: Interpret) -> Value:

        cond_res = self.cond.interpret(inter)

        if cond_res:
            return self.branch_true.interpret(inter)
        else:
            return self.branch_false.interpret(inter)

    def __str__(self) -> str:
        return (f'{CHR_IF} {str(self.cond)} '
                +f'{CHR_THEN} {str(self.branch_true)}'
                +f'{CHR_ELSE} {str(self.branch_false)}')

    def get_free_names(self) -> Set[str]:
        res = self.cond.get_free_names()
        res.update(self.branch_true.get_free_names())
        res.update(self.branch_false.get_free_names())
        return res

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
        -> Response[Value]:

        if stage.stage == 0:
            return Dependency(1, [self.cond], None)
        if stage.stage == 1:
            cond = stage.args[0]
            return Dependency(2,
                              [self.branch_true if cond else self.branch_false],
                              None)
        if stage.stage == 2:
            return stage.args[0]

        assert False


Atom = Union[FunctionApplication,
             Constant,
             FunctionDefinition,
             SequenceDefinition,
             Identifier,
             IfStmt]


Expression = Atom


class Assignment(AstElement[Tuple[str, Value]]):

    def __init__(self, location: Location, name: str, expr: Expression):
        super().__init__(location)
        self.name = name
        self.expr = expr

    def __str__(self) -> str:
        return self.name + f' {CHR_ASSIGN} ' + str(self.expr)

    def interpret(self, inter: Interpret) -> Tuple[str, Value]:

        binding : FutureBinding[Value] = FutureBinding()
        inter.sstack.put_on_last_future(self.name, binding)
        value = ast_interpret(inter, self.expr)
        binding.value = value
        inter.sstack.put_on_last(self.name, value)
        return self.name, value

    def get_free_names(self) -> Set[str]:
        return self.expr.get_free_names()


class Document(AstElement[Dict[str, Value]]):

    def __init__(self, location: Location, *assignments: Assignment) -> None:
        super().__init__(location)
        self.assignments = assignments

    def __str__(self) -> str:
        return '\n'.join(str(a) for a in self.assignments)

    def interpret(self, inter: Interpret) -> Dict[str, Value]:
        res: Dict[str, Value] = {}
        for assignment in self.assignments:
            key, value = assignment.interpret(inter)
            res[key] = value
        return res

    def get_free_names(self) -> Set[str]:
        res = set()
        for a in self.assignments:
            res.update(a.get_free_names())
        return res

###############################################################################
# Parsing
###############################################################################

def match_token(state: ParsingState[Lexem, Any],
                *lex_id: int) -> Optional[Lexem]:
    return state.match_pred(lambda x: x[0] in lex_id)

def perror_msg(location: Tuple[int, int], message: str) -> str:
    return f'{str_of_location(location)} - {message}'

def perror_lex_msg(lex: Lexem, message: str) -> str:
    return perror_msg(lex[2], message)

def perror_expected(lex: Lexem, *expected: str) -> str:
    return perror_lex_msg(lex,
                          "expected " + ', '.join(f"{e}" for e in expected)
                          + f" but got '{lex[1]}'")

def parsing_error(location: Tuple[int, int], message: str) -> ParseError:
    return ParseError(perror_msg(location, message))


def req_token(state: ParsingState[Lexem, Any], *lex_id: int) -> Lexem:

    head = None
    head = state.rpop()

    if head[0] not in lex_id:
        raise ParseError(
            perror_expected(head, *(str_of_lexid(i) for i in lex_id)))
    return head

def peek_token(state: ParsingState[Lexem, Any]) -> Optional[int]:
    lexem = state.peek()
    if lexem is None:
        return None
    return lexem[0]


def cur_location(state: ParsingState[Lexem, Grammar]) -> Location:
    return state.rpeek()[2]

def parse_document(state: ParsingState[Lexem, Grammar]) -> Document:
    location = cur_location(state)
    res: List[Assignment] = []
    while state and state.peek()[0] != LEX_EOF:
        if peek_token(state) in {LEX_OPBINL, LEX_OPBINR}:
            res.append(parse_operator_definition(state))
        else:
            res.append(parse_assignment(state))
    req_token(state, LEX_EOF)
    return Document(location, *res)


def parse_operator_definition(state: ParsingState[Lexem, Grammar]) -> Assignment:
    location = cur_location(state)
    op_def = req_token(state, LEX_OPBINL, LEX_OPBINR)
    operator = req_token(state, LEX_OPERATOR)
    level = req_token(state, LEX_LIT_INT)
    function = parse_expression(state)
    req_token(state, LEX_SEMICOLON);

    level_int = int(level[1])

    if level_int < 0 or level_int >= state.data.operator_levels:
        raise parsing_error(location, f'operator level out of bounds {level_int}')

    state.data.add_operator(operator[1], level_int, 2, op_def[0] == LEX_OPBINR)

    return Assignment(location, operator[1], function)


def parse_assignment(state: ParsingState[Lexem, Grammar]) -> Assignment:
    location = cur_location(state)
    name = req_token(state, LEX_IDENTIFIER)
    req_token(state, LEX_ASSIGN)
    expr = parse_expression(state)
    req_token(state, LEX_SEMICOLON)
    return Assignment(location, name[1], expr)


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
    return Identifier(token[2], token[1])


def parse_function_definition(state: ParsingState[Lexem, Grammar]) \
        -> FunctionDefinition:

    location = cur_location(state)


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
    return FunctionDefinition(location, arguments, expr, producing=producing)


def parse_sequence(state: ParsingState[Lexem, Grammar]) -> Expression:

    location = cur_location(state)

    req_token(state, LEX_LBRACK)

    elements = []

    while match_token(state, LEX_RBRACK) is None:
        elements.append(parse_expression(state))
        if (match_token(state, LEX_COMMA) is None):
            break

    req_token(state, LEX_RBRACK)

    return SequenceDefinition(location, elements)

def parse_if_stmt(state: ParsingState[Lexem, Grammar]) -> IfStmt:

    location = cur_location(state)

    req_token(state, LEX_IF)
    cond = parse_expression(state)
    req_token(state, LEX_THEN)
    branch_true = parse_expression(state)
    req_token(state, LEX_ELSE)
    branch_false = parse_expression(state)
    return IfStmt(location, cond, branch_true, branch_false)

LITERALS = [LEX_LIT_STR, LEX_LIT_DOUBLE, LEX_LIT_INT]

def parse_literal(state: ParsingState[Lexem, Grammar]) -> Constant:
    token = req_token(state, *LITERALS)
    if (token[0] == LEX_LIT_STR):
        return Constant(token[2], str(token[1]))
    if (token[0] == LEX_LIT_DOUBLE):
        return Constant(token[2], float(token[1]))
    if (token[0] == LEX_LIT_INT):
        return Constant(token[2], int(token[1]))
    assert False

def parse_atom(state: ParsingState[Lexem, Grammar]) -> Expression:

    token = state.peek()
    if (token is None):
        raise PatternReject(f"Invalid token: {state.peek()}")
    if (token[0] == LEX_LPARE):
        state.rpop()
        expr = parse_expression(state)
        req_token(state, LEX_RPARE)
        return expr
    if (token[0] in LITERALS):
        return parse_literal(state)
    if (token[0] == LEX_FUN):
        return parse_function_definition(state)
    if (token[0] == LEX_LBRACK):
        return parse_sequence(state)
    if (token[0] == LEX_IDENTIFIER):
        return parse_identifier(state)
    if (token[0] == LEX_IF):
        return parse_if_stmt(state)

    raise PatternReject(perror_expected(token, 'expression element'))


def match_operator(state: ParsingState[Lexem, Grammar], operators: Iterable[str]):

    if not state:
        return None

    lex = state.rpeek()

    if lex[0] != LEX_OPERATOR:
        return None

    if lex[1] in operators:
        state.rpop()
        return Identifier(lex[2], lex[1])

    if lex[1] not in state.data.operators:
        raise ParseError(
            perror_lex_msg(lex,
                            f'operator {lex[1]} not defined'))



def parse_expression_level_unary(state: ParsingState[Lexem, Grammar],
                                 level: int) -> Expression:

    _, _, prefix, posfix = state.data.operator_table[level - 1]
    prefix_tokens = [(LEX_OPERATOR, op) for op in prefix]
    posfix_tokens = [(LEX_OPERATOR, op) for op in posfix]

    prefix_stack = []

    while (op := match_operator(state, prefix)) is not None:
        prefix_stack.append(op)

    body = parse_expression_level(state, level - 1)

    for prefix_operator in reversed(prefix_stack):
        body = FunctionApplication(prefix_operator, body)

    while (op := match_operator(state, posfix)) is not None:
        body = FunctionApplication(op, body)

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


    while (op := match_operator(state, infix)) is not None:
        operators.append(op)
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


def stage_function(fun: Callable[..., Value]) -> F[Value]:
    def stage_function_w(inter: Interpret, stage: CurrentStage) -> Value:
        return fun(inter, *stage.args)
    return stage_function_w


def builtin_function(inter: Interpret, name: str, argument_count: int) \
        -> Callable[[Callable[..., Value]], Callable[..., Value]]:
    def builtin_function_d(fun: Callable[..., Value]) -> Callable[..., Value]:
        define_builtin(
            inter,
            name,
            FunctionObject(stage_function(fun), argument_count))
        return fun
    return builtin_function_d


def builtin_operator(inter: Interpret, grammar: Grammar, name: str, level: int,
                     arity: int, associativity: bool) \
        -> Callable[[Callable[..., Value]], Callable[..., Value]]:
    assert arity in {1, 2}

    def builtin_function_d(fun: Callable[..., Value]) -> Callable[..., Value]:
        define_builtin(
            inter,
            name,
            FunctionObject(stage_function(fun), arity))
        grammar.add_operator(name, level, arity, associativity)
        return fun
    return builtin_function_d


###############################################################################
# Application
###############################################################################

grammar = Grammar(5)

inter = Interpret()
inter.sstack.add_scope()

def type_check(a: Value, b) -> None:
    if not isinstance(a, b):
        raise TypeError(f'{a} is not of type {b}')


@builtin_operator(inter, grammar, '+', 1, 2, False)
def op_plus(_: Interpret, a: Value, b: Value) -> int:
    type_check(a, int)
    type_check(b, int)
    return a + b


@builtin_operator(inter, grammar, '*', 1, 2, False)
def op_mul(_: Interpret, a: Value, b: Value) -> int:
    type_check(a, int)
    type_check(b, int)
    return a * b


@builtin_operator(inter, grammar, '-', 1, 2, False)
def op_minus(_: Interpret, a: Value, b: Value) -> int:
    type_check(a, int)
    type_check(b, int)
    return a - b


@builtin_operator(inter, grammar, '=', 0, 2, False)
def op_eq(_: Interpret, a: Value, b: Value) -> bool:
    return a == b

def load_document(filename: str) -> Document:
    state = ParsingState(gen_of_file(filename), None, '\n')
    lex_state = ParsingState((lexer(state)), grammar)
    return parse_document(lex_state)

try:
    res = load_document('../sandbox/prelude.sq').interpret(inter)
    res.update(load_document('../sandbox/test.sq').interpret(inter))
    print(res['main'])
except (ParseError, PatternReject, InterError) as e:
    print(str(e))
