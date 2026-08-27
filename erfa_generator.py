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
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from pathlib import Path
from string import Template
from typing import Final, final

DEFAULT_ERFA_LOC = Path(__file__).with_name("liberfa") / "erfa" / "src"
DEFAULT_TEMPLATE_LOC = Path(__file__).with_name("erfa")


class FunctionDoc:
    def __init__(self, doc: str, pyname: str) -> None:
        self.pyname: Final = pyname
        if pyname == "ldn":
            doc = doc.removeprefix("+")
        elif pyname == "aticqn":
            doc = doc.replace("\n* ", "\n** ", 2).replace("\n*\n", "\n**\n", 1)
        self.doc: Final = re.sub(
            r"^\*\* {,2}", "", doc.removeprefix("\n"), flags=re.MULTILINE
        )

        get_arg_doc_list = functools.partial(
            self._get_arg_doc_list, n_spaces=2 if pyname in ("ab", "refco") else 3
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
        if m := re.search(r"[- ]+\n\n(.+?\.)\s", self.doc, re.DOTALL):
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

    def inner_loop_steps_and_copy(self, name_suffix: str = "") -> list[str]:
        if self.signature_shape == "()":
            return []
        name = self.name + name_suffix
        lines = [f"npy_intp is_{name}{i} = *steps++;" for i in range(self.ndim)]
        # copy should be made if buffer not contiguous;
        # note: one can only have 1 or 2 dimensions
        if self.ndim == 1:
            lines.append(f"int copy_{name} = (is_{name}0 != sizeof({self.ctype}));")
        else:
            lines.extend([
                f"int copy_{name} = (is_{name}1 != sizeof({self.ctype}) ||",
                f"          is_{name}0 != {self.shape[1]} * sizeof({self.ctype}));",
            ])
        return lines

    @functools.cached_property
    def cast_pointer(self) -> str:
        return f"_{self.name} = (({self.ctype} (*){self.cshape}){self.name});"

    def copy_elements(self, direction: str, name_suffix: str = "") -> str:
        name = self.name + name_suffix
        shape_description = "".join(str(n) for n in self.shape if n is not None)
        func_name = f"copy_{direction}_{self.ctype}{shape_description}"
        args = [name, *[f"is_{name}{i}" for i in range(self.ndim)], self.name_for_call]
        return _assemble_func_call(func_name, args) + ";"


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
        self.descriptons: Final = {
            "else" if code == "else" else int(code): " ".join(
                line.strip() for line in description.splitlines()
            )
            for code, description in re.findall(
                r"(-?\w+) = ((?:[^=]+$)+)", status.group(1), re.MULTILINE
            )
            if code != "0"
        }

    def to_python(self) -> list[str]:
        return ["{", *[f'    {k!r}: "{v}",' for k, v in self.descriptons.items()], "}"]


class Return(Variable):
    pass


class ResultTuple:
    def __init__(self, func_name: str, args: Iterable[Argument | Return]) -> None:
        self.name: Final = f"{func_name.capitalize()}Result"
        self.args: Final = tuple(args)

    def create(self) -> str:
        return _assemble_func_call(self.name, [arg.name for arg in self.args])

    def define(self) -> str:
        arg_names = ", ".join(arg.name for arg in self.args)
        return f'{self.name} = namedtuple("{self.name}", "{arg_names}")'


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
        templateloc: Path,
    ) -> None:
        self.name: Final = name
        self.pyname: Final = name.removeprefix("era").lower()
        self.doc: Final = doc
        self.c_retval: Final = c_retval
        self.templateloc: Final = templateloc

        self.in_args: Final = tuple(a for a in args if a.name in self.doc.input)
        self.inout_args: Final = tuple(a for a in args if a.name in self.doc.inout)
        self.out_args: Final = tuple(a for a in args if a.name in self.doc.output)

        self.py_args: Final = (*self.in_args, *self.inout_args)
        self.c_args: Final = (*self.py_args, *self.out_args)

    @classmethod
    def from_c_code(cls, name: str, source_path: Path, templateloc: Path) -> "Function":
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
            UFunc(name, doc, args, c_retval, templateloc)
            if all(arg.signature_shape == "()" for arg in args)
            else GUFunc(name, doc, args, c_retval, templateloc)
        )

    @functools.cached_property
    def py_return(self) -> Argument | Return | ResultTuple:
        returns: tuple[Argument | Return, ...] = (
            (*self.inout_args, *self.out_args, self.c_retval)
            if isinstance(self.c_retval, Return)
            else (*self.inout_args, *self.out_args)
        )
        return returns[0] if len(returns) == 1 else ResultTuple(self.pyname, returns)

    @functools.cached_property
    def ufunc_return(self) -> tuple[Variable, ...]:
        return (
            (*self.inout_args, *self.out_args)
            if self.c_retval is None
            else (*self.inout_args, *self.out_args, self.c_retval)
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

    def generate_python_body(self) -> list[str]:
        ufunc_name = f"ufunc.{self.pyname}"
        arg_names = [arg.name for arg in self.py_args]
        lines = [
            _assemble_func_call(
                ufunc_name,
                in_args=arg_names,
                out_args=[arg.name for arg in self.ufunc_return],
            )
        ]
        if isinstance(self.c_retval, StatusCode) and self.c_retval.descriptons:
            lines.append(f'check_errwarn({self.c_retval.name}, "{self.pyname}")')
        lines.extend(
            f"{arg.name} = {arg.name}.view(dt_bytes1)"
            for arg in self.out_args
            if arg.ctype == "char"
        )
        if len(lines) == 1 and not isinstance(self.c_retval, StatusCode):
            ufunc_call = f"{ufunc_name}({', '.join(arg_names)})"
            return [
                f"return {self.py_return.name}(*{ufunc_call})"
                if isinstance(self.py_return, ResultTuple)
                else f"return {ufunc_call}"
            ]
        lines.append(
            f"return {self.py_return.create()}"
            if isinstance(self.py_return, ResultTuple)
            else f"return {self.py_return.name}"
        )
        return lines

    @property
    def init_ufunc_loop_local_vars(self) -> list[str]:
        lines: list[str] = []
        for f in ("char *{} = *args++;".format, "npy_intp s_{} = *steps++;".format):
            lines.extend(f(arg.name) for arg in self.in_args)
            lines.extend(f(arg.name + "_in") for arg in self.inout_args)
            lines.extend(f(arg.name) for arg in self.ufunc_return)
        for arg in self.c_args:
            if arg.signature_shape == "()" or arg.ctype == "eraLDBODY":
                lines.append(f"{arg.ctype} (*_{arg.name}){arg.cshape};")
            else:
                lines.extend([
                    f"double b_{arg.name}{arg.cshape};",
                    f"{arg.ctype} (*_{arg.name}){arg.cshape} = &b_{arg.name};",
                ])
        if self.c_retval:
            lines.append(f"{self.c_retval.ctype} _{self.c_retval.name};")
        return lines

    @functools.cached_property
    def ufunc_loop_template(self) -> Template:
        return Template((self.templateloc / "ufunc_loop.templ").read_text().strip())

    @functools.cached_property
    def ufunc_loop(self) -> str:
        arg_pointer_incrementation = ", ".join(
            [f"{arg.name} += s_{arg.name}" for arg in self.in_args + self.ufunc_return]
            + [f"{arg.name}_in += s_{arg.name}_in" for arg in self.inout_args],
        )
        return self.ufunc_loop_template.substitute(
            pyname=self.pyname,
            init_ufunc_loop_local_vars=_indent(self.init_ufunc_loop_local_vars),
            increment_arg_pointers=arg_pointer_incrementation,
            ufunc_inner_loop_body=_indent(self.ufunc_loop_inner_loop_body, 2),
        )

    @property
    def ufunc_loop_inner_loop_body(self) -> list[str]:
        lines = [*[a.cast_pointer for a in self.c_args if a.signature_shape == "()"]]
        for arg in filter(lambda a: a.signature_shape == "()", self.inout_args):
            size = 1
            for s in arg.shape:
                if s is None:
                    raise RuntimeError(
                        f"{arg.name} size in {self.name} not known at compile-time"
                    )
                size *= s
            lines.extend([
                f"if ({arg.name}_in != {arg.name}) {{",
                f"    memcpy({arg.name}, {arg.name}_in, {size}*sizeof({arg.ctype}));",
                "}",
            ])
        call = _assemble_func_call(self.name, [a.name_for_call for a in self.c_args])
        if retval := self.c_retval:
            lines.extend([
                f"_{retval.name} = {call};",
                f"*(({retval.ctype} *){retval.name}) = _{retval.name};",
            ])
        else:
            lines.append(call + ";")
        return lines

    @property
    def define_types_and_functions(self) -> list[str]:
        if self.user_dtype:
            return []
        # for non-structured functions, define there types and functions
        # as these do not get copied
        npy_types = [arg.npy_type for arg in self.py_args + self.ufunc_return]
        return [
            f"static char types_{self.pyname}[{len(npy_types)}] = {{{', '.join(npy_types)}}};",
            f"static PyUFuncGenericFunction funcs_{self.pyname}[1] = {{ &ufunc_loop_{self.pyname} }};",
        ]

    @functools.cached_property
    def define_ufunc(self) -> str:
        placeholders = {
            "name": self.name,
            "pyname": self.pyname,
            "n_py_args": len(self.py_args),
            "n_ufunc_return": len(self.ufunc_return),
            "signature": self.signature,
        }
        if self.user_dtype:
            placeholders["user_dtype"] = self.user_dtype
            placeholders["register_dtypes"] = "\n".join(
                f"dtypes[{i}] = {arg.dtype};"
                for i, arg in enumerate(self.py_args + self.ufunc_return)
            )
            file = "define_ufunc_user_dtype.templ"
        else:
            file = "define_ufunc.templ"
        return Template((self.templateloc / file).read_text()).substitute(placeholders)

    @functools.cached_property
    def py_docstring(self) -> str:
        lines = ['"""', self.doc.first_sentence]
        if self.py_args:
            lines.extend(_docstring_section_title("Parameters"))
            lines.extend(f"{arg.name} : {arg.ctype} array" for arg in self.py_args)
        lines.extend(_docstring_section_title("Returns"))
        if isinstance(self.py_return, ResultTuple):
            lines.append(
                f"A ``{self.py_return.name}`` namedtuple with the following attributes:"
            )
            lines.extend(f"{a.name} : {a.ctype} array" for a in self.py_return.args)
        else:
            lines.append(f"{self.py_return.name} : {self.py_return.ctype} array")
        lines.extend(_docstring_section_title("Notes"))
        lines.append(f"Wraps ERFA function ``{self.name}``. ")
        if inout_names := ", ".join(arg.name for arg in self.inout_args):
            lines[-1] += "Note that, unlike the erfa routine,"
            lines.append(
                f"the python wrapper does not change {inout_names} in-place. ",
            )
        lines[-1] += "The ERFA documentation is::\n"
        lines.extend([textwrap.indent(self.doc.doc, 4 * " "), '"""'])
        return "\n".join(lines)

    @functools.cached_property
    def to_python(self) -> str:
        wrapper = _indent([
            f"def {self.pyname}({', '.join(arg.name for arg in self.py_args)}):",
            *self.py_docstring.splitlines(),
            *self.generate_python_body(),
        ])
        return (
            f"{self.py_return.define()}\n\n\n{wrapper}"
            if isinstance(self.py_return, ResultTuple)
            else wrapper
        )

    @functools.cached_property
    def ufunc_signature(self) -> str:
        param_types = ", ".join(f"{arg.name}: Any" for arg in self.py_args)
        return_type = (
            "Any"
            if len(self.ufunc_return) == 1
            else f"tuple[{', '.join('Any' for arg in self.ufunc_return)}]"
        )
        return f"def {self.pyname}({param_types}) -> {return_type}: ..."


class UFunc(Function):
    @functools.cached_property
    def signature(self) -> str:
        return "NULL"


class GUFunc(Function):
    @functools.cached_property
    def signature(self) -> str:
        return (
            f'"{",".join(arg.signature_shape for arg in self.py_args)}'
            f'->{",".join(arg.signature_shape for arg in self.ufunc_return)}"'
        )

    @functools.cached_property
    def ufunc_loop_template(self) -> Template:
        template_file = (
            "eraLDBODY_ufunc_loop.templ"
            if self.user_dtype == "dt_eraLDBODY"
            else "ufunc_loop.templ"
        )
        return Template((self.templateloc / template_file).read_text().strip())

    @property
    def init_ufunc_loop_local_vars(self) -> list[str]:
        lines = super().init_ufunc_loop_local_vars
        lines.extend([  # only LDBODY has non-fixed dimension; it is always first
            "int nb = (int)dimensions[0];  /* Refuse to worry about INT_MAX */"
            for arg in self.in_args
            if arg.ctype == "eraLDBODY"
        ])
        for arg in self.c_args:
            if arg in self.inout_args:
                lines.extend(arg.inner_loop_steps_and_copy("_in"))
            lines.extend(arg.inner_loop_steps_and_copy())
        return lines

    @property
    def ufunc_loop_inner_loop_body(self) -> list[str]:
        lines = []
        for arg in self.c_args:
            if arg.signature_shape != "()":
                lines.extend([
                    f"if (!copy_{arg.name}) {{",
                    f"    {arg.cast_pointer}",
                    "}",
                ])
                if arg in self.in_args:  # copy input arguments to buffer if needed
                    lines.extend([
                        "else {",
                        f"    {arg.copy_elements('to')}",
                        "}",
                    ])
                elif arg in self.inout_args:
                    # for inout arguments copy to output if needed
                    lines.extend([
                        f"if (copy_{arg.name}_in || {arg.name} != {arg.name}_in) {{",
                        f"    {arg.copy_elements('to', '_in')}",
                        "}",
                    ])
        lines.extend(super().ufunc_loop_inner_loop_body)
        for arg in self.c_args:
            if arg.signature_shape != "()" and arg not in self.in_args :
                lines.extend([
                    f"if (copy_{arg.name}) {{",
                    f"    {arg.copy_elements('from')}",
                    "}",
                ])
        return lines


class Constant:

    def __init__(self, name: str, value: str, doc: str) -> None:
        self.name = name.replace("ERFA_", "")
        self.value = value.replace("ERFA_", "")
        self.doc = doc

    @functools.cached_property
    def define(self) -> str:
        return "\n".join([f"{self.name} = {self.value}", f'"""{self.doc}"""'])


class TestFunction:
    """Function holding information about a test in t_erfa_c.c"""

    def __init__(
        self, func: Function, t_erfa_c: str, erfa_funcs: Mapping[str, Function]
    ) -> None:
        self.func: Final = func
        # Get lines that test the given erfa function: capture everything
        # between a line starting with '{' after the test function definition
        # and the first line starting with '}' or ' }'.
        search = re.search(
            rf"^static void t_{func.pyname}\(.+?\).+?Called: (.+?)$.+?^\{{(.+?)^\s?\}}",
            t_erfa_c,
            re.DOTALL | re.MULTILINE,
        )
        if search is None:
            raise RuntimeError(f"cannot find the test for {func.name}")
        self.called_functions: Final[Mapping[str, Function]] = {
            name: erfa_funcs[name]
            for name in re.findall(r"(\w+),?", search.group(1))
            if name not in (self.func.name, "and", "viv", "vvd")
        }
        source = re.sub(r"\s\s+", " ", search.group(2))
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
                    (not rest and ctype != "eraASTROM")
                    or name in self.func.doc.output  # no need to initialize outputs
                    or any(
                        name == arg.name
                        for f in self.called_functions.values()
                        for arg in f.ufunc_return
                    )
                    or name == "iydmf"  # eraJdcalf test has a typo
                ):
                    continue
                if name in self.dt_pv_vars:
                    defines.append(f"{name} = np.void(None, erfa_ufunc.dt_pv)")
                    continue
                v = Argument(f"{ctype} {var}")
                shape = v.shape if v.signature_shape != "()" else "()"
                dtype = "float" if v.dtype == "dt_double" else "erfa_ufunc." + v.dtype
                defines.append(f"{name} = np.empty({shape}, {dtype})")
                if ctype in  ("eraASTROM", "eraLDBODY"):
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
                in_args, out_args = _args_from_func_call(line, self.func)
                if self.func.c_retval:
                    out_args.append(line.split(" =", 1)[0])
                line = _assemble_func_call(
                    f"erfa_ufunc.{self.func.pyname}", in_args, out_args
                )
                if 'astrom' in out_args:
                    out.append(line)
                    line = 'astrom = astrom.view(np.recarray)'

            # In some test functions, there are calls to other ERFA functions.
            elif called_func := self.called_functions.get(line.split("(", 1)[0]):
                in_args, out_args = _args_from_func_call(line, called_func)
                if isinstance(called_func.c_retval, StatusCode):
                    out_args.append("j")
                line = _assemble_func_call(
                    f"erfa_ufunc.{called_func.pyname}", in_args, out_args
                )

            out.append(line)

        return out


def _args_from_func_call(line: str, func: Function) -> tuple[list[str], list[str]]:
    args = [
        arg.strip().removeprefix("&")
        for arg in line.split("(", 1)[1].removesuffix(")").split(",")
    ]
    for i, elem in enumerate(func.c_args):
        if elem.ctype == "eraLDBODY":
            args.pop(i)  # pyerfa does not require array sizes as separate arguments.
    in_args = [
        # convert any C octal integer literals       [
        str(int(arg, 8)) if arg.startswith("0") and arg.isdigit() else arg
        for arg in args[: len(func.py_args)]
    ]
    return in_args, args[len(func.in_args) :]


def _assemble_func_call(
    name: str, in_args: list[str], out_args: list[str] | None = None
) -> str:
    func_call = f"{name}({', '.join(in_args)})"
    return f"{', '.join(out_args)} = {func_call}" if out_args else func_call


def _indent(lines: list[str], levels: int = 1) -> str:
    for i in range(1, len(lines)):
        if lines[i]:
            lines[i] = levels * "    " + lines[i]
    return "\n".join(lines)


def _docstring_section_title(title: str) -> tuple[str, str, str]:
    return ("", title, len(title) * "-")


def _render_template(template: Path, /, **kwargs: str) -> None:
    template.with_suffix("").write_text(
        Template(template.read_text()).substitute(**kwargs)
    )


def main(srcdir: Path, templateloc: Path) -> None:
    funcs = [
        Function.from_c_code(name, srcdir, templateloc)
        for name in re.findall(
            r"\w+ (\w+)\(.*?\);", (srcdir / "erfa.h").read_text(), flags=re.DOTALL
        )
    ]
    funcs_sorted_by_name = {f.name: f for f in sorted(funcs, key=lambda f: f.pyname)}

    constants: list[Constant] = []
    for chunk in (srcdir / "erfam.h").read_text().split("\n\n"):
        doc = "\n".join(re.findall(r"/\* (.+?) \*/\n", chunk, flags=re.DOTALL))
        constants.extend(
            Constant(name, value, doc)
            for name, value in re.findall(
                r"#define (ERFA_\w+?) \(?(.+?)\)?$",
                chunk,
                flags=re.DOTALL | re.MULTILINE,
            )
        )

    _render_template(
        templateloc / "core.py.templ",
        all_list=_indent([
            *[f'"{constant.name}",' for constant in constants],
            '"ErfaError",',
            '"ErfaWarning",',
            *[f'"{func.pyname}",' for func in funcs],
        ]),
        status_code_entries=_indent([
            f'"{func.pyname}": {_indent(scode.to_python())},'
            for func in funcs_sorted_by_name.values()
            if isinstance((scode := func.c_retval), StatusCode) and scode.descriptons
        ]),
        constants="\n".join(constant.define for constant in constants),
        funcs="\n\n\n".join([func.to_python for func in funcs]),
    )

    _render_template(
        templateloc / "ufunc.c.templ",
        ufunc_loops="\n\n".join(func.ufunc_loop for func in funcs),
        type_and_func_definitions=_indent(
            list(chain(*[func.define_types_and_functions for func in funcs]))
        ),
        ufunc_definitions=_indent(
            list(chain(*[func.define_ufunc.splitlines() for func in funcs]))
        ),
    )

    create_test_funcs = functools.partial(
        TestFunction,
        t_erfa_c=(srcdir / "t_erfa_c.c").read_text(),
        erfa_funcs=funcs_sorted_by_name,
    )
    _render_template(
        templateloc / "tests" / "test_ufunc.py.templ",
        test_functions="\n\n\n".join([
            _indent([f"def test_{tfunc.func.pyname}() -> None:", *tfunc.to_python()])
            for tfunc in map(create_test_funcs, funcs_sorted_by_name.values())
        ]),
    )

    _render_template(
        templateloc / "ufunc.pyi.templ",
        funcs="\n\n\n".join(func.ufunc_signature for func in funcs),
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
