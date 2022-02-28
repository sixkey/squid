from __future__ import annotations
from typing import (Generator, TypeVar, Tuple, Generic, Optional, Callable,
                    Union, List, Any, Dict, Set, Iterable, Type, cast)
from ppretty import ppretty
import argparse as ap
import sys

###############################################################################
# TODO
###############################################################################
"""

Syntax:
    - (operator) - ifentifier
    - coma - generate tuples

Operators:
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

IO:
    - definition of IO

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


def trace(a: T, msg: Optional[str] = None) -> T:
    if msg:
        print(f'{msg}:', a)
    else:
        print(a)

    return a

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
                    '<', '>', '=', '$', '|', '&', '!', '.', '^'))

LEX_LIT_INT = define_lex('int literal')
LEX_LIT_DOUBLE = define_lex('double literal')
LEX_LIT_STR = define_lex('str literal')
LEX_LIT_CHR = define_lex('character literal')
LEX_IDENTIFIER = define_lex('identifier')

LEX_PROD, CHR_PROD = define_keyword('=>')
LEX_FUN, CHR_FUN = define_keyword('fun')
LEX_OBJ, CHR_OBJ = define_keyword('obj')
LEX_LET, CHR_LET = define_keyword('let')
LEX_IN, CHR_IN = define_keyword('in')
LEX_TRUE, CHR_TRUE = define_keyword('true')
LEX_FALSE, CHR_FALSE = define_keyword('false')

LEX_OPBINL, CHR_OPBINL = define_keyword('opbinl')
LEX_OPBINR, CHR_OPBINR = define_keyword('opbinr')

LEX_FUN_DELIM, CHR_FUN_DELIM = define_keyword('->')
LEX_ASSIGN, CHR_ASSIGN = define_keyword(':=')
LEX_LCEOP_OSTART, CHR_LCEOP_OSTART = define_keyword('>>')
LEX_LCEOP_CSTART, CHR_LCEOP_CSTART = define_keyword('-<')
LEX_LCEOP_OEND, CHR_LCEOP_OEND = define_keyword('<<')
LEX_LCEOP_CEND, CHR_LCEOP_CEND = define_keyword('>-')
LEX_FUN_PATTERN, CHR_FUN_PATTERN = define_keyword('|-')

LEX_INTEN_BIGSEP, CHR_INTEN_BIGSEP = define_keyword('|')
LEX_INTEN_ASSIGN, CHR_INTEN_ASSIGN = define_keyword('<-')
LEX_INTEN_SEP, CHR_INTEN_SEP = define_keyword(':')

LEX_IF, CHR_IF = define_keyword('if')
LEX_THEN, CHR_THEN = define_keyword('then')
LEX_ELSE, CHR_ELSE = define_keyword('else')

LEX_EOF = define_lex('eof')
CHR_EOF = 'eof'

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
        elif identifier_start(character):
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

        if lexem is not None:
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


def identifier_start(c: str) -> bool:
    return c.isalpha() or c == '_'


def identifier_char(c: str) -> bool:
    return c.isalnum() or c == '_' or c == "'"


def lex_identifier(state: ParsingState[str, None]) -> LexemCore:
    buffer = state.req_pred(identifier_start)

    while state and identifier_char(state.rpeek()):
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
        try:
            self.value: Optional[T] = next(self.gen)
        except StopIteration:
            self.value = None

        self.row_element: Optional[T] = row_element

        self.row_counter = 0
        self.col_counter = 0

        self.name_stack: List[str] = []

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
        self.value: Optional[T] = None


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

    def pop_scope(self) -> Dict[K, Binding[V]]:
        return self.stack.pop()

    def copy(self, layers: Optional[int] = None) -> 'ScopeStack[K, V]':
        if layers is None:
            return ScopeStack(self.stack[:])
        return ScopeStack(self.stack[:layers])

    def __contains__(self, key: K) -> bool:
        return self.lookup(key) is not None

    def __str__(self) -> str:
        return 'STACK\n' + '\n'.join('\tLAYER' + ''.join(f'\n\t\t{k}: {v}' for k, v in e.items()) for e in self.stack)


###############################################################################
# Interpret state
###############################################################################

class ScopeStacks(Generic[K, V]):

    def __init__(self):
        self.stacks = []

    def add_stack(self, layers: Optional[int] = None):
        if layers:
            self.stacks.append(self.head().copy(layers))
        else:
            self.stacks.append(ScopeStack())

    def pop_stack(self):
        self.stacks.pop()

    def head(self):
        return self.stacks[-1]


class Interpret:

    def __init__(self) -> None:
        self.sstacks: ScopeStacks[str, Value] = ScopeStacks()

    def sstack(self) -> ScopeStack[str, Value]:
        return self.sstacks.head()


class InterError(BaseException):
    pass

class BultinError(BaseException):
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
        self.give_amount = give_amount if give_amount is not None else len(
            children)


class CurrentStage(Generic[T]):

    def __init__(self, stage: int, args: List[Value], data: T):
        self.stage = stage
        self.args = args
        self.data = data


Response = Union[T, Dependency[R]]


class SequenceObject:
    pass


class RunningSequenceObject:
    pass


class FieldObject(Generic[T]):

    def __init__(self, size: int, def_value: T):
        self.field = [def_value] * size

    def __str__(self) -> str:
        return '{' + ', '.join(str(e) for e in self.field) + '}'


F = Callable[[Interpret, CurrentStage], Response[T, None]]


class TypeConstructor:

    def __init__(self, elements: List[Value], name: str):
        self.elements = elements
        self.name = name

    def __str__(self) -> str:
        if self.name == 'Tuple':
            return '(' + ','.join(str(e) for e in self.elements) + ')'
        return self.name + ' '.join(str(e) for e in self.elements)


class FunctionObject:

    def __init__(self, f: F[Response[Value, None]],
                 argument_count: int, *partial: Value, info: Optional[str] = None):
        super().__init__()
        self.f = f
        self.argument_count = argument_count
        self.partial = list(partial)
        self.info: Optional[str] = info

    def copy(self, *partial: Value) -> FunctionObject:
        return FunctionObject(
            self.f, self.argument_count - len(partial), *partial, info=self.info)

    def apply(self, inter: Interpret, stage: CurrentStage[bool]) \
            -> Response[Value, None]:
        assert len(stage.args) <= self.argument_count

        force, fun_data = False, None

        if stage.data is not None:
            force, fun_data = stage.data

        if force:
            return self.f(inter,
                          CurrentStage(stage.stage, stage.args, fun_data))
        if len(stage.args) == self.argument_count:
            return self.f(inter,
                          CurrentStage(
                              stage.stage, self.partial + stage.args, fun_data))

        return atom_object(self.copy(*stage.args))

    def __repr__(self) -> str:
        return f'FunctionObject {self.argument_count}'# {self.info}'


class ObjectObject:

    def __init__(self, name: str, values: List[Value]):
        self.name = name
        self.values = values

    def __str__(self) -> str:
        return self.name + ''.join(' (' + str(v) + ')' for v in self.values)


class AtomObject(ObjectObject):

    def __init__(self, name: str, values: List[Value]):
        self.name = name
        self.hidden = values
        self.values = [self]

    def __str__(self) -> str:
        return self.name + ''.join(' ' + str(v) for v in self.hidden)


class AstSequenceObject(SequenceObject):

    def __init__(self, elements: List[CaptureExpression]):
        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def __str__(self):
        return '[ ' + ', '.join(str(e) for e in self.elements) + ' ]'


class AstSequenceObjectIterator(RunningSequenceObject):

    def __init__(self, sequence_object: AstSequenceObject):
        self.sequence_object = sequence_object
        self.pointer = 0
        self.active_child: Optional[CaptureExpression] = None

    def inherit(self, iterator: AstSequenceObjectIterator) -> None:
        self.pointer = iterator.pointer
        self.sequence_object = iterator.sequence_object

    def inter_step(self, inter: Interpret,
                   stage: CurrentStage[Optional[int]]) \
            -> Response[Value, Optional[int]]:

        if stage.stage == 0:
            if self.pointer >= len(self.sequence_object.elements):
                return Bottom

            if self.active_child is not None:
                return Dependency(2, [self.active_child], None)
            else:
                lower = self.sequence_object.elements[self.pointer].inter_step(
                    inter, CurrentStage(0, [], None))

        elif stage.stage == 1:
            assert (stage.data is not None)
            seq_stage = stage.data
            current_object = self.sequence_object.elements[self.pointer]
            lower = current_object.inter_step(
                inter, CurrentStage(seq_stage, stage.args, None))

        elif stage.stage == 2:
            lower = stage.args[0]
            if isinstance(lower, ObjectObject):
                if (lower.name == 'Bottom'):
                    self.pointer += 1
                    self.active_child = None
                    return self.inter_step(
                        inter, CurrentStage(0, [], None))
            assert lower is not None
            return lower
        else:
            assert False

        if isinstance(lower, Dependency):
            dep = Dependency(1, lower.children, lower.stage)
            return dep

        if (sequence := safe_get_atom_value(lower, AstSequenceObjectIterator)) is not None:
            self.active_child = self.sequence_object.elements[self.pointer].copy(sequence)
            return self.inter_step(inter, CurrentStage(0, [], None))
        if (sequence := safe_get_atom_value(lower, IntensionalSequenceIterator)) is not None:
            self.active_child = self.sequence_object.elements[self.pointer].copy(sequence)
            return self.inter_step(inter, CurrentStage(0, [], None))
        else:
            self.pointer += 1
            assert lower is not None
            return lower

    def __str__(self) -> str:
        return f'Iterator {self.pointer} {self.sequence_object}'


Value = Union[bool, int, float, str, FunctionObject, SequenceObject,
              ObjectObject, SequenceObject, RunningSequenceObject,
              FieldObject]

###############################################################################
# Evaluation
###############################################################################


def take(a: List[T], count: int) -> List[T]:
    res = []
    for _ in range(count):
        res.append(a.pop())
    return res


def ast_interpret(inter: Interpret, root: AstElement[Any]) -> Any:

    pile: List[Value] = []
    stack: List[Tuple[int, int, Any, AstElement[Any]]] = [(0, 0, None, root)]

    while stack:

        stage, arg_count, data, element = stack.pop()
        stage_pack = CurrentStage(stage, take(pile, arg_count), data)
        response = element.inter_step(inter, stage_pack)

        # print("STATE")
        # print(ppretty(element))
        # print(ppretty(stage_pack))
        # print(ppretty(data))
        # print(ppretty(response))

        if isinstance(response, Dependency):
            stack.append(
                (response.stage, response.give_amount, response.data, element))
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
                     associativity: int) -> None:

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

    def inter_error(self, message: str, e: type = InterError) -> InterError:
        return e(f'{str_of_location(self.location)} - {message}')

    def interpret(self, inter: Interpret) -> T:
        ...

    def get_free_names(self) -> Set[str]:
        ...

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
            -> Response[Value, Any]:
        ...


def atom_object(value: Value) -> AtomObject:

    name = None
    if isinstance(value, bool):
        name = 'Bool'
    elif isinstance(value, int):
        name = 'Int'
    elif isinstance(value, str):
        name = 'Str'
    elif isinstance(value, float):
        name = 'Double'
    elif isinstance(value, SequenceObject):
        name = 'Sequence'
    elif isinstance(value, RunningSequenceObject):
        name = 'RunningSequence'
    elif isinstance(value, FieldObject):
        name = 'Field'
    elif isinstance(value, FunctionObject):
        name = 'Fun'
    else:
        raise RuntimeError(f'atomic value is not defined for {value}')

    o = AtomObject(name, [value])
    return o


class Constant(AstElement[Value]):

    def __init__(self, location: Location, value: Value):
        super().__init__(location)
        self.value = value

    def __str__(self) -> str:
        if(isinstance(self.value, str)):
            return f"\"{self.value}\""
        return f"{self.value}"

    def interpret(self, _: Interpret) -> Value:
        return atom_object(self.value)

    def get_free_names(self) -> Set[str]:
        return set()

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
            -> Response[Value, None]:
        return atom_object(self.value)


class Identifier(AstElement[Value]):

    def __init__(self,
                 location: Location,
                 name: str):
        super().__init__(location)
        self.name = name

    def __str__(self) -> str:
        return self.name

    def interpret(self, inter: Interpret) -> Value:
        val = inter.sstack().lookup(self.name)
        if val is None:
            raise self.inter_error(f"'{self.name}' not defined")
        return val

    def get_free_names(self) -> Set[str]:
        return set([self.name])

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
            -> Response[Value, None]:
        val = inter.sstack().lookup(self.name)
        if val is None:
            print(inter.sstack())
            raise self.inter_error(f"'{self.name}' not defined")
        return val


def reverse(lst: List[T]) -> List[T]:
    return list(reversed(lst))


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

        # The application invokation
        if stage.stage == 0:
            return Dependency(1, [self.fun], None)

        # The function is evaluated
        if stage.stage == 1:

            fun_atom = stage.args[0]
            if (not isinstance(fun_atom, AtomObject)
                or not isinstance(fun_atom.hidden[0], FunctionObject)):
                raise self.inter_error(f'{fun_atom} is not a function')
            fun_val = get_atom_value(stage.args[0], FunctionObject)

            argument_count = min(len(self.arguments), fun_val.argument_count)
            # Evaluate everything, pop only argument_count for the next
            # evaluation
            return Dependency(
                2, reverse(self.arguments),
                (0, fun_val, len(self.arguments) - argument_count, False, None),
                argument_count)

        # The function evaluation
        if stage.stage == 2:

            assert (stage.data is not None)

            fun_stage, fun_val, reminder, force, fun_data = stage.data

            try:
                response = fun_val.apply(inter, CurrentStage(
                    fun_stage, reverse(stage.args), (force, fun_data)))
            except (PatternNotMatched, BultinError) as e:
                raise self.inter_error(str(e), InterError)

            # Function needs to evaluate something
            if isinstance(response, Dependency):
                return Dependency(
                    2, response.children,
                    (response.stage, fun_val, reminder, True, response.data))

            # Result is a function
            if (isinstance(response, AtomObject)
                    and isinstance(response.hidden[0], FunctionObject) and reminder != 0):
                res_fun = get_atom_value(response, FunctionObject)
                argument_delta = min(reminder, res_fun.argument_count)
                return Dependency(
                    2, [],
                    (0, res_fun, reminder - argument_delta, False, None),
                    argument_delta)

            reminder = max(reminder - fun_val.argument_count, 0)

            if reminder != 0:
                raise self.inter_error(
                    f"Function takes {len(self.arguments) - reminder} " +
                    f"arguments, but {len(self.arguments)} were given.")

            return response

        assert False


class Constructor(AstElement[ObjectObject]):

    def __init__(self, location: Location, name: str,
                 arguments: List[Expression]):
        super().__init__(location)
        self.name = name
        self.arguments = arguments

    def get_free_names(self) -> Set[str]:
        res = set()
        for arg in self.arguments:
            res.update(arg.get_free_names())
        return res

    def inter_step(self, inter: Interpret, stage: CurrentStage[Any]) \
            -> Response[Value, Any]:

        if stage.stage == 0:
            return Dependency(1, reverse(self.arguments), None)

        if stage.stage == 1:
            values = stage.args
            return ObjectObject(self.name, reverse(values))

        assert False


def list_of_opt(opt: Optional[T]) -> List[T]:
    return [] if opt is None else [opt]


class FunctionOption(AstElement[FunctionObject]):

    def __init__(self, location: Location, formal_arguments: List[Pattern],
                 expr: Expression, producing: Optional[str] = None):
        super().__init__(location)
        self.formal_arguments = formal_arguments
        self.expr = expr
        self.producing = producing
        self.capture_names = self.get_free_names()

    def __str__(self) -> str:
        return ("".join(' ' + str(s) for s in self.formal_arguments)
                + (f' {CHR_PROD} {self.producing}'
                    if self.producing is not None
                    else '')
                + f' {CHR_FUN_DELIM} '
                + str(self.expr))

    def match(self, arguments: List[Value]) -> Optional[Dict[str, Value]]:
        if len(arguments) != len(self.formal_arguments):
            return None
        res = {}
        for pattern, arg in zip(self.formal_arguments, arguments):
            if pattern_match(pattern, arg, res) is None:
                return None
        return res

    def get_free_names(self) -> Set[str]:
        names = self.expr.get_free_names()
        assert names is not None
        remove = set(list_of_opt(self.producing))
        for pattern in self.formal_arguments:
            pattern_names = pattern_get_free_names(pattern)
            remove.update(pattern_names)
        names = names.difference(remove)
        return names


class FunctionDefinition(AstElement[FunctionObject]):

    def __init__(self, location: Location, options: List[FunctionOption],
                 arity: int, producing: Optional[str] = None):
        super().__init__(location)
        self.options = options
        self.producing = producing
        self.capture_names = self.get_free_names()
        self.arity = arity

    def __str__(self) -> str:
        return (CHR_FUN
                + f' {CHR_FUN_PATTERN}'.join(str(o) for o in self.options))

    def inflate(self, other: FunctionDefinition) -> bool:
        if self.arity != other.arity:
            return False
        self.options += other.options
        return True

    def inter_step(self, inter: Interpret, stage: CurrentStage) \
            -> Response[AtomObject, Optional[FutureBinding[Value]]]:

        closure: Dict[str, Binding[Value]] = {}

        for name in self.capture_names:
            value = inter.sstack().lookup_future(name)
            if value is None:
                raise self.inter_error(f"value '{name}' not defined")
            closure[name] = value

        def f(inter: Interpret, stage: CurrentStage[None]) \
                -> Response[Value, None]:

            # Function is started
            if stage.stage == 0:

                inter.sstacks.add_stack(1)

                # closure scope
                inter.sstack().add_scope()
                for name, value in closure.items():
                    inter.sstack().put_on_last_future(name, value)

                # Scope for arguments
                inter.sstack().add_scope()

                res_binding = FutureBinding()
                if self.producing is not None:
                    inter.sstack().put_on_last_future(
                        self.producing, res_binding)

                assignments = None
                errors = []
                chosen_option = None

                for option in self.options:
                    try:
                        assignments = option.match(stage.args)
                        chosen_option = option
                        break
                    except (PatternNotMatched) as e:
                        errors.append(e)

                if assignments is None:
                    raise self.inter_error(
                        'None of the patterns matched: '
                        + ''.join('\n  ' + str(e) for e in errors), PatternNotMatched)

                for name, arg in assignments.items():
                    inter.sstack().put_on_last(name, arg)

                # scope for body
                inter.sstack().add_scope()

                return Dependency(1, [chosen_option.expr], res_binding)

            if stage.stage == 1:

                # function over, pop its stack
                inter.sstacks.pop_stack()

                if stage.data is not None:
                    res_binding = stage.data
                    res_binding.value = stage.args[0]

                return stage.args[0]

            assert False

        return atom_object(FunctionObject(f, self.arity, info=str(self)))

    def get_free_names(self) -> Set[str]:
        return set.union(*(o.get_free_names() for o in self.options))


Bottom = ObjectObject('Bottom', [])
Unit = ObjectObject('Unit', [])


class CaptureExpression:

    def __init__(self, expression: Expression, inter: Optional[Interpret] = None):
        self.expression = expression
        self.capture = {}

        if inter is not None:
            capture_names = self.expression.get_free_names()
            for name in capture_names:
                val = inter.sstack().lookup_future(name)
                if val is None:
                    raise expression.inter_error(
                        'symbol not found in outer context')
                self.capture[name] = val

    def copy(self, expression: Expression) -> CaptureExpression:
        capture = CaptureExpression(expression)
        capture.capture = self.capture
        return capture

    def inter_step(self, inter: Interpret, stage: CurrentStage) \
            -> Response[Value, None]:

        if stage.stage == 0:
            inter.sstack().add_scope()
            for name, value in self.capture.items():
                inter.sstack().put_on_last(name, value)
            inter.sstack().add_scope()
            return Dependency(
                1, [self.expression], None)

        if stage.stage == 1:
            res = stage.args[0]
            assert res is not None
            inter.sstack().pop_scope()
            inter.sstack().pop_scope()
            return res

        assert False

    def __str__(self) -> str:
        return str(self.expression)


class SequenceDefinition(AstElement[SequenceObject]):

    def __init__(self, location: Location, elements: List[Expression]):
        super().__init__(location)
        self.elements = elements

    def __str__(self) -> str:
        return (CHR_LBRACK
                + f'{CHR_COMMA} '.join(str(e) for e in self.elements)
                + CHR_RBRACK)

    def inter_step(self, inter: Interpret, stage: CurrentStage) \
            -> Response[SequenceObject, None]:
        return atom_object(AstSequenceObject(
            [CaptureExpression(e, inter) for e in self.elements]))

    def get_free_names(self):
        return set.union(
            set(), *(e.get_free_names() for e in self.elements))


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
                + f'{CHR_THEN} {str(self.branch_true)}'
                + f'{CHR_ELSE} {str(self.branch_false)}')

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
            cond = get_atom_value(stage.args[0], bool)
            return Dependency(2,
                              [self.branch_true
                               if cond
                               else self.branch_false],
                              None)
        if stage.stage == 2:
            return stage.args[0]

        assert False


class LetIn(AstElement[Value]):

    def __init__(self, location: Location, pattern: Pattern,
                 expression: Expression,
                 body: Expression):
        super().__init__(location)
        self.pattern = pattern
        self.expression = expression
        self.body = body

    def __str__(self) -> str:
        return (f'{CHR_LET} {str(self.pattern)} {CHR_ASSIGN} '
                + f'({str(self.expression)}) {CHR_IN} ({str(self.body)})')

    def inter_step(self, inter: Interpret, stage: CurrentStage[None]) \
            -> Response[Value]:

        if stage.stage == 0:
            return Dependency(1, [self.expression], None)

        if stage.stage == 1:
            assignments = {}
            pattern_match(self.pattern, stage.args[0], assignments)
            inter.sstack().add_scope()
            for key, value in assignments.items():
                inter.sstack().put_on_last(key, value)
            return Dependency(2, [self.body], None)

        if stage.stage == 2:
            inter.sstack().pop_scope()
            return stage.args[0]

        assert False

    def get_free_names(self) -> Set[str]:
        expression_names = self.expression.get_free_names()
        pattern_names = pattern_get_free_names(self.pattern)
        return expression_names - pattern_names



Atom = Union[FunctionApplication,
             Constant,
             FunctionDefinition,
             SequenceDefinition,
             Identifier,
             IfStmt,
             LetIn,
             Constructor]


Expression = Atom

class IntensionalAssignment:

    def __init__(self, pattern: Pattern, expression: Expression):
        self.pattern = pattern
        self.expression = expression

    def __str__(self):
        return str(self.pattern) + f' {CHR_INTEN_ASSIGN} ' + str(self.expression)

class IntensionalSequence(AstElement[AstSequenceObject]):

    def __init__(self, main: Expression, parts: List[Union[Expression, IntensionalAssignment]]):
        super().__init__(main.location)
        self.main = main
        self.parts = parts

    def __len__(self):
        return len(self.parts)

    def __str__(self):
        return ('*[ '  + str(self.main) + f' {CHR_INTEN_BIGSEP} '
                + CHR_INTEN_SEP.join(str(e) for e in self.parts) + ' ]')

    def get_free_names(self) -> Set[str]:

        pos = set()
        neg = set()

        for part in self.parts:
            if isinstance(part, IntensionalAssignment):
                pos.update(part.expression.get_free_names())
                neg.update(part.pattern.get_free_names())
            else:
                pos.update(part.get_free_names())

        return pos - neg;

    def inter_step(self, inter: Interpret,
                   stage: CurrentStage[None]) \
            -> Response[Value, None]:
        return atom_object(IntensionalSequenceIterator(self))

class IntensionalSequenceIterator(RunningSequenceObject):

    def __init__(self, inten_seq: IntensionalSequence):
        self.inten_seq = inten_seq

        self.pointer: int = 0

        self.values: Dict[str, Value] = {}
        self.generator_stack: List[Tuple[int, RunningSequenceObject]] = []
        self.cur_top_generator: Optional[Tuple[int, RunningSequenceObject]] = None

    def inter_step(self, inter: Interpret,
                   stage: CurrentStage[Optional[Pattern]]) \
            -> Response[Value, Optional[Pattern]]:

        # Add scope
        if stage.stage == 0:
            inter.sstack().add_scope()
            for key, value in self.values.items():
                inter.sstack().put_on_last(key, value)

        # First entry - pick what to do
        if stage.stage == 0 or stage.stage == 1:
            # There is nothing more to do

            if self.pointer == -1:
                return Dependency(5, [], None)

            if self.pointer >= len(self.inten_seq):
                return Dependency(4, [self.inten_seq.main], None)

            current = self.inten_seq.parts[self.pointer]

            if isinstance(current, IntensionalAssignment):
                if (self.cur_top_generator is not None
                        and self.pointer == self.cur_top_generator[0]):
                    # Currently running sequence
                    return Dependency(2, [self.cur_top_generator[1]], current.pattern)
                # Sequence that needs to be started
                return Dependency(2, [current.expression], current.pattern)
            # Normal value
            return Dependency(3, [current], None)

        # We need to eval a intensional assignment
        elif stage.stage == 2 or stage.stage == 6:
            value = stage.args[0]

            if (sequence := safe_get_atom_value(value, RunningSequenceObject)) is not None:
                # We see the generator for the first time, meaning
                # we need to return
                if stage.stage == 2:
                    self.generator_stack.append((self.pointer, sequence))
                return Dependency(6, [sequence], stage.data)

            if isinstance(value, ObjectObject) and value.name == 'Bottom':
                self.generator_stack.pop()
                if len(self.generator_stack) == 0:
                    return Dependency(5, [], None)
                else:
                    self.cur_top_generator = self.generator_stack[-1]
                    self.pointer = self.cur_top_generator[0]
            else:
                pattern = stage.data
                assert pattern is not None
                assignments = {}

                pattern_match(pattern, value, assignments)

                self.values.update(assignments)

                for key, value in assignments.items():
                    inter.sstack().put_on_last(key, value)

                self.pointer += 1;

            return Dependency(1, [], None)

        # We need to evaluate a predicate
        elif stage.stage  == 3:
            value = get_atom_value(stage.args[0], bool)

            if value:
                self.pointer += 1
            elif len(self.generator_stack) == 0:
                return Dependency(5, [], None)
            else:
                self.cur_top_generator = self.generator_stack[-1]
                self.pointer = self.cur_top_generator[0]

            return Dependency(1, [], None)

        # Final evaluation
        elif stage.stage == 4:
            value = stage.args[0]
            inter.sstack().pop_scope()
            assert value is not None

            if len(self.generator_stack) != 0:
                self.cur_top_generator = self.generator_stack[-1]
                self.pointer = self.cur_top_generator[0]
            else:
                self.pointer = -1

            return value

        elif stage.stage == 5:
            inter.sstack().pop_scope()
            return Bottom

        assert False

    def get_free_names(self) -> Set[str]:
        return self.inten_seq.get_free_names()


class Assignment(AstElement[Tuple[str, Value]]):

    def __init__(self, location: Location, name: str, expr: Expression):
        super().__init__(location)
        self.name = name
        self.expr = expr

    def __str__(self) -> str:
        return self.name + f' {CHR_ASSIGN} ' + str(self.expr)

    def interpret(self, inter: Interpret) -> Tuple[str, Value]:
        binding: FutureBinding[Value] = FutureBinding()
        inter.sstack().put_on_last_future(self.name, binding)
        value = ast_interpret(inter, self.expr)
        binding.value = value
        inter.sstack().put_on_last(self.name, value)
        return self.name, value

    def get_free_names(self) -> Set[str]:
        return self.expr.get_free_names()


class Document(AstElement[Dict[str, Value]]):

    def __init__(self, location: Location) -> None:
        super().__init__(location)
        self.names: Dict[str, Assignment] = {}
        self.assignments: List[Assignment] = []

    def __str__(self) -> str:
        return '\n'.join(str(a) for a in self.assignments)

    def add_definition(self, assignment: Assignment, name_stack: List[str]) -> None:
        name = assignment.name
        location = assignment.location

        if name in self.names:
            old_value = self.names[name]
            if (isinstance(old_value.expr, FunctionDefinition)
                    and isinstance(assignment.expr, FunctionDefinition)):
                if not old_value.expr.inflate(assignment.expr):
                    raise parsing_error(location, "the arities don't match", name_stack)
            else:
                raise parsing_error(location, f'redefinition of {name}', name_stack)
        else:
            self.names[name] = assignment
            self.assignments.append(assignment)

    def __iadd__(self, other: Document) -> Document:
        for assignment in other.assignments:
            self.add_definition(assignment, ['document_merging'])
        return self

    def interpret(self, inter: Interpret) -> Dict[str, Value]:
        res = {}
        for assignment in self.assignments:
            name, value = assignment.interpret(inter)
            res[name] = value
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


def token_is_next(state: ParsingState[Lexem, Any], *lex_id: int) -> bool:
    head = None
    head = state.rpeek()
    if head[0] not in lex_id:
        return False
    return True


def perror_msg(location: Tuple[int, int], message: str, name_stack: List[str]) -> str:
    return f'{str_of_location(location)} - {message} in \n' + '\n'.join(name_stack[-5:])


def perror_lex_msg(lex: Lexem, message: str, name_stack: List[str]) -> str:
    return perror_msg(lex[2], message, name_stack)


def perror_expected(lex: Lexem, name_stack: List[str], *expected: str) -> str:
    return perror_lex_msg(lex,
                          "expected " + ', '.join(f"{e}" for e in expected)
                          + f" but got '{lex[1]}'", name_stack)


def parsing_error(location: Tuple[int, int], message: str, name_stack: List[str]) -> ParseError:
    return ParseError(perror_msg(location, message, name_stack))


def req_token(state: ParsingState[Lexem, Any], *lex_id: int) -> Lexem:

    head = None
    head = state.rpop()

    if head[0] not in lex_id:
        raise ParseError(
            perror_expected(head, state.name_stack, *(str_of_lexid(i) for i in lex_id)))
    return head


def req_wholetoken(state: ParsingState[Lexem, Any], *lexes: Tuple[int, str]) \
        -> Lexem:
    head = None
    head = state.rpop()

    if (head[0], head[1]) not in lexes:
        raise ParseError(
            perror_expected(
                head, state.name_stack,
                *(f'{str_of_lexid(lex[0])} {lex[1]}' for lex in lexes)))
    return head


def peek_token(state: ParsingState[Lexem, Any]) -> Optional[int]:
    lexem = state.peek()
    if lexem is None:
        return None
    return lexem[0]


def cur_location(state: ParsingState[Lexem, Grammar]) -> Location:
    return state.rpeek()[2]


def parse_named(fun: Callable[[ParsingState[T, S]], R]) \
        -> Callable[[ParsingState[T, S]], R]:
    def wrapper(state: ParsingState[T, S]) -> R:
        # print('append:', fun.__name__)
        state.name_stack.append(fun.__name__)
        res = fun(state)
        # print('pop:', fun.__name__, 'with: ', str(res))
        state.name_stack.pop()
        return res
    wrapper.__name__ = fun.__name__
    return wrapper


@parse_named
def parse_document(state: ParsingState[Lexem, Grammar]) -> Document:

    location = cur_location(state)
    document = Document(location)

    while state and state.peek()[0] != LEX_EOF:
        if peek_token(state) in {LEX_OPBINL, LEX_OPBINR}:
            assignment = parse_operator_definition(state)
        else:
            assignment = parse_assignment(state)

        document.add_definition(assignment, state.name_stack)

        req_token(state, LEX_SEMICOLON)

    req_token(state, LEX_EOF)

    return document


@parse_named
def parse_operator_definition(state: ParsingState[Lexem, Grammar]) \
        -> Assignment:
    location = cur_location(state)
    op_def = req_token(state, LEX_OPBINL, LEX_OPBINR)
    operator = req_token(state, LEX_OPERATOR)
    level = req_token(state, LEX_LIT_INT)
    function = parse_expression(state)

    level_int = int(level[1])

    if level_int < 0 or level_int >= state.data.operator_levels:
        raise parsing_error(
            location, f'operator level out of bounds {level_int}', state.name_stack)

    state.data.add_operator(
        operator[1], level_int, 2, op_def[0] == LEX_OPBINR)

    return Assignment(location, operator[1], function)


@parse_named
def parse_assignment(state: ParsingState[Lexem, Grammar]) -> Assignment:
    location = cur_location(state)
    name = req_token(state, LEX_IDENTIFIER)
    req_token(state, LEX_ASSIGN)
    expr = parse_expression(state)
    return Assignment(location, name[1], expr)


@parse_named
def parse_letin(state: ParsingState[Lexem, Grammar]) -> LetIn:
    location = cur_location(state)
    req_token(state, LEX_LET)

    pattern = parse_pattern(state)
    req_token(state, LEX_ASSIGN)
    expression = parse_expression(state)
    req_token(state, LEX_IN)
    body = parse_expression(state)
    return LetIn(location, pattern, expression, body)


@parse_named
def parse_arguments(state: ParsingState[Lexem, Grammar]) -> List[Atom]:
    arguments: List[Expression] = []
    while True:
        try:
            arguments.append(parse_atom(state))
        except PatternReject:
            break
    return arguments


@parse_named
def parse_application(state: ParsingState[Lexem, Grammar]) -> Atom:
    fun = parse_atom(state)
    arguments = parse_arguments(state)
    if arguments == []:
        return fun
    return FunctionApplication(fun, *arguments)


@parse_named
def parse_identifier(state: ParsingState[Lexem, Grammar]) -> Identifier:
    token = req_token(state, LEX_IDENTIFIER)
    return Identifier(token[2], token[1])


class CompPattern(AstElement):

    def __init__(self, location: Location, name: str,
                 subpatterns: List[Pattern]) -> None:
        super().__init__(location)
        self.name = name
        self.subpatterns = subpatterns

    def __str__(self) -> str:
        return ('< ' + self.name
                + ''.join(' ' + str(p) for p in self.subpatterns) + ' >')

    def get_free_names(self) -> Set[str]:
        assert False


class PatternNotMatched(BaseException):
    pass


Pattern = Union[CompPattern, Identifier, Constant]


def pattern_get_free_names(pattern: Pattern) -> Set[str]:
    if isinstance(pattern, Identifier):
        return set([pattern.name])
    if isinstance(pattern, CompPattern):
        return set.union(
            set(), *(pattern_get_free_names(p) for p in pattern.subpatterns))
    if isinstance(pattern, Constant):
        return set()
    assert False, type(pattern)


def pattern_match(pattern: Pattern, value: Value, res: Dict[str, Value]) \
        -> bool:

    if isinstance(pattern, Identifier):
        res[pattern.name] = value
        return True

    if isinstance(pattern, Constant):
        if isinstance(value, AtomObject):
            if pattern.value != value.hidden[0]:
                raise pattern.inter_error(
                    f"object {value} is not a '{pattern}'", PatternNotMatched)
        else:
            if value != pattern.value:
                raise pattern.inter_error(
                    f"object {value} is not a '{pattern}'", PatternNotMatched)
        return True

    if not isinstance(value, ObjectObject):
        raise pattern.inter_error(
            f"object {value} is not a '{pattern.name}'", PatternNotMatched)

    if (value.name != pattern.name
            or len(value.values) != len(pattern.subpatterns)):
        raise pattern.inter_error(
            f"object {value} is not a '{pattern.name}'", PatternNotMatched)

    for subpattern, value_part in zip(pattern.subpatterns, value.values):
        if not pattern_match(subpattern, value_part, res):
            return False

    return True


@parse_named
def parse_pattern_atom(state: ParsingState[Lexem, Any]) -> Pattern:
    location = cur_location(state)
    if match_token(state, LEX_LPARE):
        res = parse_pattern(state)
        req_token(state, LEX_RPARE)
        return res

    if token_is_next(state, LEX_IDENTIFIER):
        return parse_identifier(state)
    if state.rpeek()[0] in LITERALS:
        return parse_literal(state)
    next_token = state.peek()
    if (next_token[0], next_token[1]) == (LEX_OPERATOR, '<'):
        state.pop()
        name = parse_identifier(state).name
        subpatterns = []
        while True:
            try:
                subpatterns.append(parse_pattern_atom(state))
            except PatternReject:
                break
        req_wholetoken(state, (LEX_OPERATOR, '>'))
        return CompPattern(location, name, subpatterns)

    raise PatternReject(
        f'token {str_of_lexid(peek_token(state))} is not a match')


@parse_named
def parse_pattern(state: ParsingState[Lexem, Any]) -> Pattern:

    location = cur_location(state)
    units = [parse_pattern_atom(state)]

    while match_token(state, LEX_COMMA):
        units.append(parse_pattern_atom(state))

    if len(units) == 1:
        return units[0]

    return CompPattern(location, 'Tuple', units)


@parse_named
def parse_function_definition(state: ParsingState[Lexem, Grammar]) \
        -> FunctionDefinition:

    location = cur_location(state)

    req_token(state, LEX_FUN)

    match_token(state, LEX_FUN_PATTERN)

    options: List[FunctionOption] = []

    arity = None

    while True:
        option_location = cur_location(state)
        arguments: List[Pattern] = []
        while True:
            try:
                arguments.append(parse_pattern(state))
            except PatternReject:
                break

        if arity is None:
            arity = len(arguments)
        elif arity != len(arguments):
            raise parsing_error(
                location, 'different options have different arrities', state.name_stack)

        producing = None
        if match_token(state, LEX_PROD):
            identifier = parse_identifier(state)
            if identifier is not None:
                producing = identifier.name
        req_token(state, LEX_FUN_DELIM)
        expr = parse_expression(state)

        options.append(FunctionOption(option_location, arguments, expr,
                                      producing=producing))

        if not match_token(state, LEX_FUN_PATTERN):
            break

    return FunctionDefinition(location, options, arity, producing=producing)


@parse_named
def parse_inten_assignment(state: ParsingState[Lexem, Grammar]) -> IntensionalAssignment:
    pattern = parse_pattern(state)
    req_token(state, LEX_INTEN_ASSIGN)
    expression = parse_expression(state)
    return IntensionalAssignment(pattern, expression)

@parse_named
def parse_sequence(state: ParsingState[Lexem, Grammar]) -> Expression:

    location = cur_location(state)

    req_token(state, LEX_LBRACK)

    elements = []

    while match_token(state, LEX_RBRACK) is None:

        expr = parse_expression(state)

        if match_token(state, LEX_INTEN_BIGSEP) is not None:

            inten_elements = []

            while not token_is_next(state, LEX_SEMICOLON, LEX_RBRACK):

                if match_token(state, LEX_LET) is not None:
                    inten_elements.append(parse_inten_assignment(state))
                else:
                    inten_elements.append(parse_expression(state))

                if match_token(state, LEX_INTEN_SEP) is None:
                    break

            elements.append(IntensionalSequence(expr, inten_elements))
        else:
            elements.append(expr)

        if (match_token(state, LEX_SEMICOLON) is None):
            break

    req_token(state, LEX_RBRACK)

    return SequenceDefinition(location, elements)


@parse_named
def parse_if_stmt(state: ParsingState[Lexem, Grammar]) -> IfStmt:

    location = cur_location(state)

    req_token(state, LEX_IF)
    cond = parse_expression(state)
    req_token(state, LEX_THEN)
    branch_true = parse_expression(state)
    req_token(state, LEX_ELSE)
    branch_false = parse_expression(state)
    return IfStmt(location, cond, branch_true, branch_false)


@parse_named
def parse_obj_definition(state: ParsingState[Lexem, Grammar]) -> Constructor:

    location = cur_location(state)

    req_token(state, LEX_OBJ)
    name = parse_identifier(state).name
    args = parse_arguments(state)

    return Constructor(location, name, args)


LITERALS = [LEX_LIT_STR, LEX_LIT_DOUBLE, LEX_LIT_INT, LEX_TRUE, LEX_FALSE]


@parse_named
def parse_literal(state: ParsingState[Lexem, Grammar]) -> Constant:
    token = req_token(state, *LITERALS)
    if (token[0] == LEX_LIT_STR):
        return Constant(token[2], str(token[1]))
    if (token[0] == LEX_LIT_DOUBLE):
        return Constant(token[2], float(token[1]))
    if (token[0] == LEX_LIT_INT):
        return Constant(token[2], int(token[1]))
    if (token[0] == LEX_TRUE):
        return Constant(token[2], True)
    if (token[0] == LEX_FALSE):
        return Constant(token[2], False)
    assert False


@parse_named
def parse_atom(state: ParsingState[Lexem, Grammar]) -> Expression:

    token = state.peek()
    if (token is None):
        raise PatternReject(f"Invalid token: {state.peek()}")
    if (token[0] == LEX_LPARE):
        state.rpop()
        expr = parse_expression(state)
        req_token(state, LEX_RPARE)
        return expr
    if (token[0] == LEX_LET):
        return parse_letin(state)
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
    if (token[0] == LEX_OBJ):
        return parse_obj_definition(state)

    raise PatternReject(perror_expected(token, state.name_stack, 'expression element'))


def match_operator(state: ParsingState[Lexem, Grammar],
                   operators: Iterable[str]):

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
                           f'operator {lex[1]} not defined', state.name_stack))


def parse_expression_level_unary(state: ParsingState[Lexem, Grammar],
                                 level: int) -> Expression:

    _, _, prefix, posfix = state.data.operator_table[level]

    prefix_stack = []

    while (op := match_operator(state, prefix)) is not None:
        prefix_stack.append(op)

    body = parse_expression_level(state, level + 1)

    for prefix_operator in reversed(prefix_stack):
        body = FunctionApplication(prefix_operator, body)

    while (op := match_operator(state, posfix)) is not None:
        body = FunctionApplication(op, body)

    return body


def parse_expression_level(state: ParsingState[Lexem, Grammar],
                           level: int) -> Expression:

    if level == len(state.data.operator_table):
        atom = parse_application(state)
        return atom

    associativity, infix, _, _ = state.data.operator_table[level]

    elements = [parse_expression_level_unary(state, level)]
    operators = []

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


@parse_named
def parse_expression(state: ParsingState[Lexem, Grammar]) -> Expression:
    return parse_tuple(state)


@parse_named
def parse_tuple(state: ParsingState[Lexem, Grammar]) -> Expression:

    location = cur_location(state)
    units = [parse_expression_unit(state)]

    while match_token(state, LEX_COMMA):
        units.append(parse_expression_unit(state))

    if len(units) == 1:
        return units[0]

    return Constructor(location, 'Tuple', units)


@parse_named
def parse_expression_unit(state: ParsingState[Lexem, Grammar]) -> Expression:

    root_level = 0

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
    inter.sstack().put(name, atom_object(val))


def stage_function(fun: Callable[..., Value]) -> F[Value]:
    def stage_function_w(inter: Interpret, stage: CurrentStage) -> Value:
        return fun(inter, *stage.args)
    return stage_function_w


def builtin_function_staged(inter: Interpret, name: str, argument_count: int) \
        -> Callable[[Callable[..., Response]], Callable[..., Value]]:
    def builtin_function_d(fun: Callable[..., Response]) \
            -> Callable[..., Value]:
        define_builtin(
            inter,
            name,
            FunctionObject(fun, argument_count))
        return fun
    return builtin_function_d


def builtin_function(inter: Interpret, name: str, argument_count: int) \
        -> Callable[[Callable[..., Value]], Callable[..., Value]]:
    def builtin_function_d(fun: Callable[..., Value]) -> Callable[..., Value]:
        define_builtin(
            inter,
            name,
            FunctionObject(stage_function(fun), argument_count))
        return fun
    return builtin_function_d


def builtin_atom_function(inter: Interpret, name: str, argument_count: int):
    def builtin_function_d(fun: Callable[..., Value]) -> Callable[..., Value]:
        def fun_wrap(inter: Interpret, *atoms: AtomObject):
            return atom_object(fun(inter, *(a.hidden[0] for a in atoms)))
        define_builtin(
            inter,
            name,
            FunctionObject(stage_function(fun_wrap), argument_count))
        return fun_wrap
    return builtin_function_d


def builtin_operator(inter: Interpret, grammar: Grammar, name: str,
                     level: int, arity: int, associativity: bool) \
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

grammar = Grammar(10)

inter = Interpret()
inter.sstacks.add_stack()
inter.sstack().add_scope()


def type_check(a: Value, b) -> None:
    if not isinstance(a, b):
        raise TypeError(f'{a} is not of type {b}')


def type_check_atom(a: Any, t: type) -> None:
    assert isinstance(a, AtomObject)
    type_check(a.hidden[0], t)


def safe_get_atom_value(a: Any, t: Type[T]) -> Optional[T]:
    if not isinstance(a, AtomObject):
        return None
    value = a.hidden[0]
    if not isinstance(value, t):
        return None
    return cast(t, value)

def get_atom_value(a: Any, t: Type[T]) -> T:
    assert isinstance(a, AtomObject)
    value = a.hidden[0]
    return cast(T, value)


@builtin_function(inter, '_field', 2)
def f_field(_: Interpret, a: AtomObject, b: Value) -> AtomObject:
    return atom_object(FieldObject(get_atom_value(a, int), b))


@builtin_function(inter, '_set', 3)
def f_set(_: Interpret, a: AtomObject, b: AtomObject, c: Value) -> Value:
    field_object = get_atom_value(a, FieldObject)
    index = get_atom_value(b, int)
    if index >= len(field_object.field):
        raise BultinError(f'field index {index} out of bounds {field_object}')
    field_object.field[index] = c
    return a


@builtin_function(inter, '_get', 2)
def f_get(inter: Interpret, a: FieldObject[Value], b: int) -> Value:
    field_object = get_atom_value(a, FieldObject)
    index = get_atom_value(b, int)
    if index >= len(field_object.field):
        raise BultinError(f'field index {index} out of bounds {field_object}')
    return field_object.field[index]


@builtin_function(inter, 'print', 1)
def f_print(_: Interpret, a: Value) -> Value:
    print(str(a))
    return a


@builtin_atom_function(inter, '_iter', 1)
def f_iter(_: Interpret, a: AstSequenceObject) -> Value:
    type_check(a, AstSequenceObject)
    return AstSequenceObjectIterator(a)


@builtin_function_staged(inter, '_next', 1)
def f_next(inter: Interpret, stage: CurrentStage) \
        -> Response[Value, Optional[Tuple[AstSequenceObjectIterator, int]]]:

    if stage.stage == 0:
        sequence = get_atom_value(stage.args[0], AstSequenceObjectIterator)
        return Dependency(1, [sequence], None)
    elif stage.stage == 1:
        res = stage.args[0]
        assert res is not None
        return res

    assert False


@builtin_atom_function(inter, '_add', 2)
def op_plus(_: Interpret, a: int, b: int) -> int:
    return a + b


@builtin_atom_function(inter, '_concat', 2)
def op_concat(_: Interpret, a: str, b: str) -> str:
    return a + b


@builtin_function(inter, '_show', 1)
def op_show(_: Interpret, a: Any) -> str:
    if isinstance(a, str):
        return f'"{a}"'
    return str(a)


@builtin_atom_function(inter, '_mul', 2)
def op_mul(_: Interpret, a: int, b: int) -> int:
    return a * b


@builtin_atom_function(inter, '_div', 2)
def op_div(_: Interpret, a: int, b: int) -> int:
    return a // b


@builtin_atom_function(inter, '_mod', 2)
def op_mod(_: Interpret, a: int, b: int) -> int:
    return a % b


@builtin_atom_function(inter, '_sub', 2)
def op_minus(_: Interpret, a: int, b: int) -> int:
    return a - b


@builtin_atom_function(inter, '_eq', 2)
def op_eq(_: Interpret, a: Any, b: Any) -> bool:
    return a == b


@builtin_atom_function(inter, '_lt', 2)
def op_lt(_: Interpret, a: Any, b: Any) -> bool:
    return a < b


def load_document(filename: str) -> Document:
    state = ParsingState(gen_of_file(filename), None, '\n')
    lex_state = ParsingState((lexer(state)), grammar)
    return parse_document(lex_state)


def parse_args(args: List[str]) -> Any:
    parser = ap.ArgumentParser()
    parser.add_argument('source_files', nargs='+')
    parser.add_argument('-r', '--raw', action='store_const',
                        const=True)
    parser.add_argument('-a', '--ast', action='store_const',
                        const=True)
    parser.add_argument('-o', '--operators', action='store_const',
                        const=True)
    return parser.parse_args(args)


try:

    arguments = parse_args(sys.argv[1:])

    res = dict()

    doc = Document((0, 0))

    if not arguments.raw:
        doc += load_document('../sandbox/prolog.sq')

    for path in arguments.source_files:
        doc += load_document(path)

    doc += load_document('../sandbox/epilog.sq')

    if arguments.operators:
        print('\n'.join(f'{i}: {o}' for i,
                        o in enumerate(grammar.operator_table)))

    if arguments.ast:
        print(doc)
    else:
        res = doc.interpret(inter)

        if '_main' not in res:
            raise InterError('_main not found')

        print(res['_main'])
except (ParseError, PatternReject, InterError, PatternNotMatched) as e:
    print(f'{e.__class__.__name__}: {str(e)}')
