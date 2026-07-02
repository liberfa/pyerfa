# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module's main purpose is to act as a script to create new versions
of ufunc.c when ERFA is updated (or this generator is enhanced).

Note that this does *not* currently automate the process of creating structs
or dtypes for those structs.  They should be added manually in the template file.
"""

import functools
import re
import textwrap
from abc import ABC, abstractproperty
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, final

from jinja2 import Environment, FileSystemLoader

DEFAULT_ERFA_LOC = Path(__file__).with_name("liberfa") / "erfa" / "src"
DEFAULT_TEMPLATE_LOC = Path(__file__).with_name("erfa")


class FunctionDoc:
    def __init__(self, doc: str, pyname: str) -> None:
        self.pyname: Final = pyname
        if pyname == "ldn":
            doc = doc.removeprefix("+")
        elif pyname == "aticqn":
            doc = doc.replace("\n* ", "\n** ", 2).replace("\n*\n", "\n**\n", 1)
        self.doc: Final = doc.replace("\n**", "\n").removeprefix("\n")

        get_arg_doc_list = functools.partial(
            self._get_arg_doc_list, n_spaces=4 if pyname in ("ab", "refco") else 5
        )
        self.input: Final = get_arg_doc_list("Given.*?\n(.+?)\n\n")
        self.inout: Final = get_arg_doc_list("Given and returned:\n(.+?)\n\n")
        self.output: Final = get_arg_doc_list("Returned.*?\n(.+?)\n\n")
        if pyname in ("aper", "aper13"):
            self.input.remove("astrom")
            self.inout.add(self.output.pop())

    def _get_arg_doc_list(self, regex: str, n_spaces: int) -> set[str]:
        """Parse input/output doc section lines, getting arguments from them.

        Also remove the nb argument in front of eraLDBODY, as we infer nb from
        the python array.
        """
        result = re.search(regex, self.doc, re.DOTALL)
        if result is None:
            return set()
        doc_list: list[str] = []
        for name, c_type in re.findall(
            rf"^{n_spaces * ' '}([\w\*,]+) +([\w\[\]\*]+) +.+?",
            result.group(1),
            re.MULTILINE,
        ):
            if c_type.startswith("eraLDBODY"):
                # Special-case LDBODY: for those, the previous argument
                # is always the number of bodies, but we don't need it
                # as an input argument for the ufunc since we're going
                # to determine this from the array itself.
                doc_list.pop()
            doc_list.extend(name.replace("*", "").split(","))
        return set(doc_list)

    @property
    def first_sentence(self) -> str:
        if m := re.search(r"[- ]+\n\n  (.+?\.)\s", self.doc, re.DOTALL):
            return m.group(1)
        raise RuntimeError(
            f"cannot find the first sentence of {self.pyname} doc comment"
        )


class Variable:
    """Properties shared by Argument, Return and StatusCode."""

    def __init__(self, ctype: str, name: str | None = None) -> None:
        self.ctype: Final = ctype
        self.name: Final = "c_retval" if name is None else name

    @final
    @property
    def npy_type(self) -> str:
        """Predefined type used by numpy ufuncs to indicate a given ctype.

        Eg., NPY_DOUBLE for double.
        """
        return "NPY_" + self.ctype.upper()

    @property
    def dtype(self) -> str:
        return "dt_" + self.ctype

    @property
    def signature_shape(self) -> str:
        return "()"

    def init_pointer_and_step_size(self, name_suffix: str = "") -> str:
        name = self.name + name_suffix
        return "\n".join([
            f"char *{name} = *args++;",
            f"npy_intp s_{name} = *steps++;",
        ])


class Argument(Variable):
    def __init__(self, definition: str) -> None:
        ctype, ptr_name_arr = definition.strip().rsplit(" ", 1)
        self.is_ptr: Final = ptr_name_arr.startswith("*")
        self.shape: Final = tuple(
            int(s) if s else None for s in re.findall(r"\[(\d*)\]", ptr_name_arr)
        )
        super().__init__(ctype, ptr_name_arr.removeprefix("*").split("[", 1)[0])

    @property
    def name_for_call(self) -> str:
        """How the argument should be used in the call to the ERFA function.

        This takes care of ensuring that inputs are passed by value,
        as well as adding back the number of bodies for any LDBODY argument.
        The latter presumes that in the ufunc inner loops, that number is
        called 'nb'.
        """
        if self.ctype == "eraLDBODY":
            return "nb, _" + self.name
        return ("_" if self.is_ptr else "*_") + self.name

    @property
    def dtype(self) -> str:
        """Name of dtype corresponding to the ctype.

        Specifically,
        double : dt_double
        int : dt_int
        double[3]: dt_vector
        double[2][3] : dt_pv
        double[2] : dt_pvdpv
        double[3][3] : dt_matrix
        int[4] : dt_ymdf | dt_hmsf | dt_dmsf, depding on name
        eraASTROM: dt_eraASTROM
        eraLDBODY: dt_eraLDBODY
        char : dt_sign
        char[] : dt_type

        The corresponding dtypes are defined in ufunc.c, where they are
        used for the loop definitions.  In core.py, they are also used
        to view-cast regular arrays to these structured dtypes.
        """
        match self.ctype, self.shape:
            case "const char", _:
                return "dt_type"
            case "char", _:
                return "dt_sign"
            case "int", (4,):
                return "dt_" + self.name[1:]
            case "double", (3,) | (3, 3):
                return "dt_double"
            case "double", (2, 3):
                return "dt_pv"
            case "double", (2,):
                return "dt_pvdpv"
            case (_, ()) | ("eraLDBODY", _):
                return super().dtype
        raise ValueError(f"ctype {self.ctype} with shape {self.shape} not recognized.")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def cshape(self) -> str:
        elems = []
        for s in self.shape:
            if s is None:
                return ""
            elems.append(f"[{s}]")
        return "".join(elems)

    @property
    def signature_shape(self) -> str:
        match self.ctype, self.shape:
            case "eraLDBODY", _:
                return "(n)"
            case "double", (3,):
                return "(3)"
            case "double", (3, 3):
                return "(3, 3)"
        return super().signature_shape

    @functools.cached_property
    def cast_pointer_and_possible_contiguous_buffer(self) -> str:
        return (
            f"{self.ctype} (*_{self.name}){self.cshape};"
            if self.signature_shape == "()" or self.ctype == "eraLDBODY"
            else "\n".join([
                f"double b_{self.name}{self.cshape};",
                f"{self.ctype} (*_{self.name}){self.cshape} = &b_{self.name};",
            ])
        )

    def inner_loop_steps_and_copy(self, name_suffix: str = "") -> str | None:
        if self.signature_shape == "()":
            return None
        name = self.name + name_suffix
        lines = [f"npy_intp is_{name}{i} = *steps++;" for i in range(self.ndim)]
        # copy should be made if buffer not contiguous;
        # note: one can only have 1 or 2 dimensions
        lines.append(
            f"int copy_{name} = (is_{name}0 != sizeof({self.ctype}));"
            if self.ndim == 1
            else (
                f"int copy_{name} = (is_{name}1 != sizeof({self.ctype}) ||\n"
                f"          is_{name}0 != {self.shape[1]} * sizeof({self.ctype}));"
            )
        )
        return "\n".join(lines)

    @functools.cached_property
    def cast_pointer(self) -> str:
        return f"_{self.name} = (({self.ctype} (*){self.cshape}){self.name});"

    @functools.cached_property
    def cast_pointer_if_needed(self) -> str:
        return "\n".join(
            [
                f"if (!copy_{self.name}) {{",
                f"    {self.cast_pointer}",
                "}",
            ]
        )

    def copy_elements(self, direction: str, name_suffix: str = "") -> str:
        name = self.name + name_suffix
        shape_description = "".join(str(n) for n in self.shape if n is not None)
        func_name = f"copy_{direction}_{self.ctype}{shape_description}"
        args = [name, *[f"is_{name}{i}" for i in range(self.ndim)], self.name_for_call]
        return _assemble_func_call(func_name, args) + ";"

    @functools.cached_property
    def memcpy_if_needed(self) -> str:
        size = 1
        for s in self.shape:
            if s is None:
                raise RuntimeError("{self.name} size not known at compile-time")
            size *= s
        return (
            f"if ({self.name}_in != {self.name}) {{\n"
            f"    memcpy({self.name}, {self.name}_in, {size}*sizeof({self.ctype}));\n"
            "}"
        )


class StatusCode(Variable):
    def __init__(self, ctype: str, doc: FunctionDoc, funcname: str) -> None:
        super().__init__(ctype)

        status = re.search(
            r"Returned \(function value\):\n\s+\w+\s+status.*?:(.+?)\s+Notes?:",
            doc.doc,
            re.DOTALL,
        )
        if status is None:
            raise RuntimeError(
                f"cannot find status code description in {funcname} doc comment"
            )
        self._statuscodes: Final = {
            "else" if code == "else" else int(code): " ".join(
                line.strip() for line in description.splitlines()
            )
            for code, description in re.findall(
                r"(-?\w+) = ((?:[^=]+$)+)", status.group(1), re.MULTILINE
            )
        }
        self.can_fail: Final = list(self._statuscodes) != [0]

    def to_python(self) -> str:
        return "\n".join(
            ["{", *[f'    {k!r}: "{v}",' for k, v in self._statuscodes.items()], "}"]
        )


class Return(Variable):
    pass


class ResultTuple:
    def __init__(self, func_name: str, args: Iterable[Argument | Return]) -> None:
        self.name: Final = f"{func_name.capitalize()}Result"
        self.arg_names: Final = ", ".join(arg.name for arg in args)

    def create(self) -> str:
        return f"{self.name}({self.arg_names})"

    def define(self) -> str:
        return f'{self.name} = namedtuple("{self.name}", "{self.arg_names}")'


class Function(ABC):
    """
    A class representing a C function.

    Parameters
    ----------
    name : str
        The name of the function
    source_path : pathlib.Path
        Directory with the file containing the function implementation.
    """

    def __init__(
        self,
        name: str,
        doc: FunctionDoc,
        args: Sequence[Argument],
        c_retval: Return | StatusCode | None,
    ) -> None:
        self.name: Final = name
        self.pyname: Final = name.removeprefix("era").lower()
        self.doc: Final = doc
        self.c_retval: Final = c_retval

        self.in_args: Final = tuple(a for a in args if a.name in self.doc.input)
        self.inout_args: Final = tuple(a for a in args if a.name in self.doc.inout)
        self.out_args: Final = tuple(a for a in args if a.name in self.doc.output)

        self.py_args: Final = (*self.in_args, *self.inout_args)
        self.c_args: Final = (*self.py_args, *self.out_args)
        self.inout_or_out_args: Final = (*self.inout_args, *self.out_args)

    @classmethod
    def from_c_code(cls, name: str, source_path: Path) -> "Function":
        pyname = name.removeprefix("era").lower()
        file = source_path / f"{pyname}.c"
        search = re.search(
            rf"(\w+) {name} ?\((.+?)\).+?/\*(.+?)\*/", file.read_text(), re.DOTALL
        )
        if search is None:
            raise RuntimeError(f"cannot find {name}() definition in {file}")

        doc = FunctionDoc(search.group(3), pyname)
        args = [Argument(arg) for arg in re.findall("[^,]+", search.group(2))]
        c_retval = None
        if (ret := search.group(1)) != "void":
            c_retval = (
                StatusCode(ret, doc, name)
                if ret == "int" and pyname not in ("tpors", "tporv")
                else Return(ret)
            )
        return (
            UFunc(name, doc, args, c_retval)
            if all(arg.signature_shape == "()" for arg in args)
            else GUFunc(name, doc, args, c_retval)
        )

    @functools.cached_property
    def result_tuple(self) -> ResultTuple | None:
        return (
            ResultTuple(self.pyname, self.py_return)
            if len(self.py_return) > 1
            else None
        )

    @functools.cached_property
    def py_return(self) -> tuple[Argument | Return, ...]:
        return (
            (*self.inout_or_out_args, self.c_retval)
            if isinstance(self.c_retval, Return)
            else self.inout_or_out_args
        )

    @functools.cached_property
    def ufunc_return(self) -> tuple[Variable, ...]:
        return (
            self.inout_or_out_args
            if self.c_retval is None
            else (*self.inout_or_out_args, self.c_retval)
        )

    @property
    def user_dtype(self) -> str | None:
        """The non-standard dtype, if any, needed by this function's ufunc.

        This would be any structured array for any input or output, but
        we give preference to LDBODY, since that also decides that the ufunc
        should be a generalized ufunc.
        """
        user_dtype = None
        for arg in self.c_args:
            if arg.ctype == 'eraLDBODY':
                return arg.dtype
            if user_dtype is None and arg.dtype not in ("dt_double", "dt_int"):
                user_dtype = arg.dtype

        return user_dtype

    @abstractproperty
    def signature(self) -> str:
        """Possible signature, if this function should be a gufunc."""

    def generate_python_body(self) -> str:
        ufunc_name = f"ufunc.{self.pyname}"
        arg_names = [arg.name for arg in self.py_args]
        lines = [
            _assemble_func_call(
                ufunc_name,
                in_args=arg_names,
                out_args=[arg.name for arg in self.ufunc_return],
            )
        ]
        if isinstance(self.c_retval, StatusCode) and self.c_retval.can_fail:
            lines.append(f'check_errwarn({self.c_retval.name}, "{self.pyname}")')
        lines.extend(
            f"{arg.name} = {arg.name}.view(dt_bytes1)"
            for arg in self.out_args
            if arg.ctype == "char"
        )
        if len(lines) == 1 and not isinstance(self.c_retval, StatusCode):
            ufunc_call = f"{ufunc_name}({', '.join(arg_names)})"
            ret_val = (
                ufunc_call
                if self.result_tuple is None
                else f"{self.result_tuple.name}(*{ufunc_call})"
            )
            return f"return {ret_val}"
        ret_val = (
            self.py_return[0].name
            if self.result_tuple is None
            else self.result_tuple.create()
        )
        lines.append(f"return {ret_val}")
        return "\n".join(lines)

    @functools.cached_property
    def init_ufunc_loop_local_vars(self) -> str:
        lines = [
            "npy_intp i_o;",  # loop index
            "npy_intp n_o = *dimensions++;",  # loop length
            *[arg.init_pointer_and_step_size() for arg in self.in_args],
            *[arg.init_pointer_and_step_size("_in") for arg in self.inout_args],
            *[arg.init_pointer_and_step_size() for arg in self.ufunc_return],
            *[arg.cast_pointer_and_possible_contiguous_buffer for arg in self.c_args],
        ]
        if self.c_retval:
            lines.append(f"{self.c_retval.ctype} _{self.c_retval.name};")
        return "\n".join(lines)

    @functools.cached_property
    def ufunc_loop_body(self) -> str:
        arg_pointer_incrementation = ", ".join(
            [f"{arg.name} += s_{arg.name}" for arg in self.in_args + self.ufunc_return]
            + [f"{arg.name}_in += s_{arg.name}_in" for arg in self.inout_args],
        )
        return "\n".join([
            self.init_ufunc_loop_local_vars,
            "for (i_o = 0; i_o < n_o;",
            f"     i_o++, {arg_pointer_incrementation}) {{",
            textwrap.indent(self.ufunc_loop_inner_loop_body, 4 * " "),
            "}",
        ])

    @abstractproperty
    def prepare_for_call(self) -> str:
        pass

    @functools.cached_property
    def ufunc_loop_inner_loop_body(self) -> str:
        lines = [self.prepare_for_call]
        call = _assemble_func_call(self.name, [a.name_for_call for a in self.c_args])
        if retval := self.c_retval:
            lines.extend([
                f"_{retval.name} = {call};",
                f"*(({retval.ctype} *){retval.name}) = _{retval.name};",
            ])
        else:
            lines.append(call + ";")
        return "\n".join(lines)


class UFunc(Function):
    @functools.cached_property
    def signature(self) -> str:
        return "NULL"

    @functools.cached_property
    def prepare_for_call(self) -> str:
        return "\n".join([
            *[arg.cast_pointer for arg in self.c_args],
            *[arg.memcpy_if_needed for arg in self.inout_args],
        ])


class GUFunc(Function):
    @functools.cached_property
    def signature(self) -> str:
        return (
            f'"{",".join(arg.signature_shape for arg in self.py_args)}'
            f'->{",".join(arg.signature_shape for arg in self.ufunc_return)}"'
        )

    @functools.cached_property
    def ufunc_loop_body(self) -> str:
        lines = [super().ufunc_loop_body]
        if self.user_dtype == "dt_eraLDBODY":
            lines.extend([
                "if (copy_b) {",
                "    free(_b);",
                "}",
            ])
        return "\n".join(lines)

    @functools.cached_property
    def init_ufunc_loop_local_vars(self) -> str:
        lines = [super().init_ufunc_loop_local_vars]
        lines.extend([  # only LDBODY has non-fixed dimension; it is always first
            "int nb = (int)dimensions[0];  /* Refuse to worry about INT_MAX */"
            for arg in self.in_args
            if arg.ctype == "eraLDBODY"
        ])
        in_only = [a.inner_loop_steps_and_copy() for a in self.in_args]
        inout = [a.inner_loop_steps_and_copy("_in") for a in self.inout_args]
        out = [a.inner_loop_steps_and_copy() for a in self.inout_or_out_args]
        lines.extend(filter(None, in_only + inout + out))
        if self.user_dtype == "dt_eraLDBODY":
            # if needed, allocate memory for contiguous eraLDBODY copies
            lines.extend([
                "if (copy_b) {",
                "    // Note that we can't use PyArray_malloc here as it is an alias to PyMem_RawMalloc",
                "    // which is not available in the Python limited API",
                "    _b = malloc(nb * sizeof(eraLDBODY));",
                "    if (_b == NULL) {",
                "        PyErr_NoMemory();",
                "        return;",
                "    }",
                "}",
                "else {",  # just to keep compiler happy
                "    _b = NULL;",
                "}",
            ])
        return "\n".join(lines)

    @functools.cached_property
    def prepare_for_call(self) -> str:
        lines = []
        for arg in self.in_args:  # copy input arguments to buffer if needed
            if arg.signature_shape == "()":
                lines.append(arg.cast_pointer)
            else:
                lines.extend([
                    arg.cast_pointer_if_needed,
                    "else {",
                    f"    {arg.copy_elements('to')}",
                    "}",
                ])
        # for inout arguments, set up output first, and then copy to it if needed
        for arg in self.inout_args:
            lines.extend(
                [arg.cast_pointer, arg.memcpy_if_needed]
                if arg.signature_shape == "()"
                else [
                    arg.cast_pointer_if_needed,
                    f"if (copy_{arg.name}_in || {arg.name} != {arg.name}_in) {{",
                    f"    {arg.copy_elements('to', '_in')}",
                    "}",
                ]
            )
        lines.extend([  # set up gufunc outputs
            a.cast_pointer if a.signature_shape == "()" else a.cast_pointer_if_needed
            for a in self.out_args
        ])
        return "\n".join(lines)

    @functools.cached_property
    def ufunc_loop_inner_loop_body(self) -> str:
        lines = [super().ufunc_loop_inner_loop_body]
        for arg in self.inout_or_out_args:
            if arg.signature_shape != "()":
                lines.extend([
                    f"if (copy_{arg.name}) {{",
                    f"    {arg.copy_elements('from')}",
                    "}",
                ])
        return "\n".join(lines)


class Constant:

    def __init__(self, name: str, value: str, doc: list[str]) -> None:
        self.name = name.replace("ERFA_", "")
        self.value = value.replace("ERFA_", "")
        self.doc = doc


class TestFunction:
    """Function holding information about a test in t_erfa_c.c"""

    def __init__(self, func: Function, t_erfa_c: str) -> None:
        self.func: Final = func
        # Get lines that test the given erfa function: capture everything
        # between a line starting with '{' after the test function definition
        # and the first line starting with '}' or ' }'.
        search = re.search(
            rf"^static void t_{func.pyname}\(" + r".+?^\{(.+?)^\s?\}",
            t_erfa_c,
            re.DOTALL | re.MULTILINE,
        )
        if search is None:
            raise RuntimeError(f"cannot find the test for {func.name}")
        source = re.sub(r"\s\s+", " ", search.group(1))
        self.definitions: Final = []
        self.lines: Final = []
        for line in re.findall(r" (.*?);", source):
            if line.startswith(("double", "int", "char", "eraASTROM", "eraLDBODY")):
                self.definitions.append(line.split(" ", 1))
            else:
                self.lines.append(line)
        self.dt_pv_vars: Final = frozenset(re.findall(r"(\w+)\[2\]\[3\]", source))

    def process_definitions(self) -> list[str]:
        defines = []
        for ctype, variables in self.definitions:
            if variables != (
                numbers := variables.removeprefix("xyz[] = {").removesuffix("}")
            ):  # Complete hack for single occurrence.
                defines.append(f"xyz = np.array([{numbers}])")
                continue
            for var in variables.split(", "):
                if "=" in var:  # only happens for double
                    defines.append(var)
                # Is variable an array?
                name, _, rest = var.partition("[")
                if (
                    not rest
                    or name in self.func.doc.output  # no need to initialize outputs
                    or name == "iydmf"  # eraJdcalf test has a typo
                ):
                    continue
                # Temporarily create an Argument, so we can use its attributes.
                # This translates, e.g., double pv[2][3] to dtype dt_pv.
                v = Argument(f"{ctype} {var}")
                shape = v.shape if v.signature_shape != "()" else "()"
                dtype = "float" if v.dtype == "dt_double" else "erfa_ufunc." + v.dtype
                defines.append(f"{name} = np.empty({shape}, {dtype})")
                if ctype == "eraLDBODY":
                    # Special case, since this should be recarray for access similar
                    # to C struct.
                    defines[-1] += ".view(np.recarray)"
        return defines

    def to_python(self) -> list[str]:
        """Lines defining the body of a python version of the test function."""
        # TODO: this is quite hacky right now!  Would be good to let function
        # calls be understood by the Function class.

        out_array_elems = tuple(f"{arg}[" for arg in self.func.doc.output)
        out = self.process_definitions()
        for line in self.lines:
            if (
                # No need to initialize output arrays in Python
                line.startswith(out_array_elems)
                # In ldn ufunc, the number of bodies is inferred from the array size,
                # so no need to keep the definition.
                or (line == "n = 3" and self.func.pyname == "ldn")
            ):
                continue

            # Actual function. Start with basic replacements.
            line = (line
                    .replace('ERFA_', 'erfa.')
                    .replace('(void)', '')
                    .replace('(int)', '')
                    .replace("pv[0]", "pv['p']")
                    .replace("pv[1]", "pv['v']")
                    .replace("s, '-'", "s[0], b'-'")  # Rather hacky...
                    .replace("s, '+'", "s[0], b'+'")  # Rather hacky...
                    .strip())

            if m := re.match(r"viv ?\( ?([\w\[\]]+), (.+?),", line):
                line = f"assert {m.group(1)} == {m.group(2)}"

            elif m := re.match(
                r"vvd\( ?(.+) ?, ([\d\.e-]+), ?([\d\.e-]+), .+?, .+?, status\)", line
            ):
                expr = m.group(1).replace(
                    self.func.name, f"erfa_ufunc.{self.func.pyname}"
                )
                line = f"assert {expr} == pytest.approx({m.group(2)}, abs={m.group(3)})"

            # Call of function that is being tested.
            elif self.func.name in line:
                # correct for LDBODY (complete hack!)
                line = line.replace('3, b', 'b').replace('n, b', 'b')
                name, arguments = _get_funcname_and_args(
                    line, self.func.name, f"erfa_ufunc.{self.func.pyname}"
                )
                args = [
                    str(int(arg, 8))  # convert any C octal integer literals
                    if arg.startswith("0") and arg.isdigit()
                    else arg.removeprefix("&")
                    for arg in arguments
                ]
                out_args = args[len(self.func.in_args) :]
                # If the call assigned something, that will have been the status.
                # Prepend any arguments assigned in the call.
                if " = " in name:
                    status, name = name.split(" = ", 1)
                    out_args.append(status)
                line = _assemble_func_call(
                    name, args[: len(self.func.py_args)], out_args
                )
                if 'astrom' in out_args:
                    out.append(line)
                    line = 'astrom = astrom.view(np.recarray)'

            # In some test functions, there are calls to other ERFA functions.
            # Deal with those in a super hacky way for now.
            elif line.startswith('eraA'):
                name, args = _get_funcname_and_args(line, "eraA", "erfa_ufunc.a")
                line = _assemble_func_call(
                    name,
                    in_args=[arg for arg in args if "&" not in arg],
                    out_args=[arg.replace("&", "") for arg in args if "&" in arg],
                )
                if 'atioq' in line or 'atio13' in line or 'apio13' in line:
                    line = line.replace(' =', ', j =')

            # And the same for some other functions, which always have a
            # 2-element time as inputs.
            elif line.startswith('eraS'):
                name, args = _get_funcname_and_args(line, "eraS", "erfa_ufunc.s")
                line = _assemble_func_call(name, in_args=args[:2], out_args=args[2:])

            # Input number setting.
            elif '=' in line:
                # Hack to make astrom element assignment work.
                if line.startswith('astrom'):
                    out.append('astrom = np.zeros((), erfa_ufunc.dt_eraASTROM).view(np.recarray)')
                # Change access to p and v elements for double[2][3] pv arrays
                # that were not caught by the general replacement above (e.g.,
                # with names not equal to 'pv')
                name, _, rest = line.partition('[')
                if name in self.dt_pv_vars and rest.startswith(("0", "1")):
                    line = name + "[" + ("'p'" if rest[0] == "0" else "'v'") + rest[1:]

            out.append(line)

        return out


def _get_funcname_and_args(
    line: str, c_prefix: str, py_prefix: str
) -> tuple[str, list[str]]:
    funcname, args = line.replace(c_prefix, py_prefix).split("(", 1)
    return funcname, [arg.strip() for arg in args.removesuffix(")").split(",")]


def _assemble_func_call(
    name: str, in_args: list[str], out_args: list[str] | None = None
) -> str:
    func_call = f"{name}({', '.join(in_args)})"
    return f"{', '.join(out_args)} = {func_call}" if out_args else func_call


def main(srcdir: Path, templateloc: Path) -> None:
    env = Environment(loader=FileSystemLoader(templateloc))

    funcs = [
        Function.from_c_code(name, srcdir)
        for name in re.findall(
            r"\w+ (\w+)\(.*?\);", (srcdir / "erfa.h").read_text(), flags=re.DOTALL
        )
    ]

    constants: list[Constant] = []
    for chunk in (srcdir / "erfam.h").read_text().split("\n\n"):
        doc = re.findall(r"/\* (.+?) \*/\n", chunk, flags=re.DOTALL)
        constants.extend(
            Constant(name, value, doc)
            for name, value in re.findall(
                r"#define (ERFA_\w+?) \(?(.+?)\)?$",
                chunk,
                flags=re.DOTALL | re.MULTILINE,
            )
        )

    outfn = "core.py"
    (templateloc / outfn).write_text(
        env.get_template(outfn + ".templ").render(funcs=funcs, constants=constants)
    )

    ufuncfn = "ufunc.c"
    (templateloc / ufuncfn).write_text(
        env.get_template(ufuncfn + ".templ").render(funcs=funcs)
    )

    testloc = templateloc / "tests"
    testfn = "test_ufunc.py"
    create_test_funcs = functools.partial(
        TestFunction, t_erfa_c=(srcdir / "t_erfa_c.c").read_text()
    )
    (testloc / testfn).write_text(
        Environment(loader=FileSystemLoader(testloc))
        .get_template(testfn + ".templ")
        .render(
            test_funcs=sorted(map(create_test_funcs, funcs), key=lambda f: f.func.name)
        )
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument(
        "srcdir",
        default=DEFAULT_ERFA_LOC,
        nargs="?",
        help=(
            "Directory where the ERFA c and header files can be found. "
            f'Default: "{DEFAULT_ERFA_LOC}"'
        ),
    )
    ap.add_argument('-t', '--template-loc',
                    default=DEFAULT_TEMPLATE_LOC,
                    help='the location where the "core.py.templ" and '
                         '"ufunc.c.templ templates can be found.')
    args = ap.parse_args()
    main(Path(args.srcdir), Path(args.template_loc))
