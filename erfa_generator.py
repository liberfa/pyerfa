# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module's main purpose is to act as a script to create new versions
of ufunc.c when ERFA is updated (or this generator is enhanced).

`Jinja2 <http://jinja.pocoo.org/>`_ must be installed for this
module/script to function.

Note that this does *not* currently automate the process of creating structs
or dtypes for those structs.  They should be added manually in the template file.
"""

import functools
import re
from pathlib import Path
from typing import Final

DEFAULT_ERFA_LOC = Path(__file__).with_name("liberfa") / "erfa" / "src"
DEFAULT_TEMPLATE_LOC = Path(__file__).with_name("erfa")


class FunctionDoc:

    def __init__(self, doc):
        self.doc = doc.replace("**", "      ").replace("/*\n", "").replace("*/", "")
        self.doc = self.doc.replace("/*+\n", "")        # accommodate eraLdn
        self.doc = self.doc.replace("*  ", "    " * 2)  # accommodate eraAticqn
        self.doc = self.doc.replace("*\n", "\n")        # accommodate eraAticqn

    def _get_arg_doc_list(self, doc_lines):
        """Parse input/output doc section lines, getting arguments from them.

        Ensure all elements of eraASTROM and eraLDBODY are left out, as those
        are not input or output arguments themselves.  Also remove the nb
        argument in from of eraLDBODY, as we infer nb from the python array.
        """
        doc_list = []
        skip = []
        for d in doc_lines:
            arg_doc = ArgumentDoc(d)
            if arg_doc.name is not None:
                if skip:
                    if skip[0] == arg_doc.name:
                        skip.pop(0)
                        continue
                    raise RuntimeError(
                        f"We whould be skipping {skip[0]} but {arg_doc.name} encountered."
                    )

                if arg_doc.type.startswith('eraLDBODY'):
                    # Special-case LDBODY: for those, the previous argument
                    # is always the number of bodies, but we don't need it
                    # as an input argument for the ufunc since we're going
                    # to determine this from the array itself. Also skip
                    # the description of its contents; those are not arguments.
                    doc_list.pop()
                    skip = ['bm', 'dl', 'pv']
                elif arg_doc.type.startswith('eraASTROM'):
                    # Special-case ASTROM: need to skip the description
                    # of its contents; those are not arguments.
                    skip = ['pmt', 'eb', 'eh', 'em', 'v', 'bm1',
                            'bpn', 'along', 'xpl', 'ypl', 'sphi',
                            'cphi', 'diurab', 'eral', 'refa', 'refb']

                doc_list.append(arg_doc)

        return doc_list

    @functools.cached_property
    def input(self):
        input_ = []
        for regex in (
            "Given([^\n]*):.*?\n(.+?)  \n",
            "Given and returned([^\n]*):\n(.+?)  \n",
        ):
            result = re.search(regex, self.doc, re.DOTALL)
            if result is not None:
                doc_lines = result.group(2).split("\n")
                input_ += self._get_arg_doc_list(doc_lines)
        return input_

    @functools.cached_property
    def output(self):
        output = []
        for regex in (
            "Given and returned([^\n]*):\n(.+?)  \n",
            "Returned([^\n]*):.*?\n(.+?)  \n",
        ):
            result = re.search(regex, self.doc, re.DOTALL)
            if result is not None:
                doc_lines = result.group(2).split("\n")
                output += self._get_arg_doc_list(doc_lines)
        return output

    @property
    def title(self):
        # Used for the docstring title.
        lines = [line.strip() for line in self.doc.split('\n')[4:10]]
        # Always include the first line, then stop at either an empty
        # line or at the end of a sentence.
        description = lines[:1]
        for line in lines[1:]:
            if line == '':
                break
            if '. ' in line:
                line = line[:line.index('. ')+1]
            description.append(line)
            if line.endswith('.'):
                break

        return '\n    '.join(description)

    def __repr__(self):
        return '\n'.join([(ln.rstrip() if ln.strip() else '')
                          for ln in self.doc.split('\n')])


class ArgumentDoc:

    def __init__(self, doc):
        if (match := re.search("^ +([^ ]+)[ ]+([^ ]+)[ ]+.+", doc)) is not None:
            self.name = match.group(1)
            if self.name.startswith('*'):  # Easier than getting the regex to behave...
                self.name = self.name.replace('*', '')
            self.type = match.group(2)
        else:
            self.name = None
            self.type = None


class Variable:
    """Properties shared by Argument, Return and StatusCode."""
    @property
    def npy_type(self):
        """Predefined type used by numpy ufuncs to indicate a given ctype.

        Eg., NPY_DOUBLE for double.
        """
        return "NPY_" + self.ctype.upper()

    @property
    def dtype(self):
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
            case _, ():
                return "dt_" + self.ctype
        raise ValueError(f"ctype {self.ctype} with shape {self.shape} not recognized.")

    @property
    def view_dtype(self):
        """Name of dtype corresponding to the ctype for viewing back as array.

        E.g., dt_double for double, dt_double33 for double[3][3].

        The types are defined in core.py, where they are used for view-casts
        of structured results as regular arrays.
        """
        if self.ctype == 'const char':
            return 'dt_bytes12'
        if self.ctype == "char":
            return 'dt_bytes1'
        raise ValueError("Only char ctype should need view back!")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        size = 1
        for s in self.shape:
            size *= s
        return size

    @property
    def cshape(self):
        return ''.join([f'[{s}]' for s in self.shape])

    @property
    def signature_shape(self):
        match self.ctype, self.shape:
            case "eraLDBODY", _:
                return "(n)"
            case "double", (3,):
                return "(3)"
            case "double", (3, 3):
                return "(3, 3)"
        return "()"


class Argument(Variable):

    def __init__(self, definition, doc):
        self.doc = doc
        self.ctype, ptr_name_arr = definition.strip().rsplit(" ", 1)
        name_arr = ptr_name_arr.removeprefix("*").removesuffix("[]")
        self.is_ptr = name_arr != ptr_name_arr
        if "[" in name_arr:
            self.name, arr = name_arr.split("[", 1)
            self.shape = tuple([int(size) for size in arr[:-1].split("][")])
        else:
            self.name = name_arr
            self.shape = ()

    @functools.cached_property
    def inout_state(self):
        inout_state = ""
        for i in self.doc.input:
            if self.name in i.name.split(","):
                inout_state = "in"
        for o in self.doc.output:
            if self.name in o.name.split(","):
                inout_state = "inout" if inout_state == "in" else "out"
        return inout_state

    @property
    def name_for_call(self):
        """How the argument should be used in the call to the ERFA function.

        This takes care of ensuring that inputs are passed by value,
        as well as adding back the number of bodies for any LDBODY argument.
        The latter presumes that in the ufunc inner loops, that number is
        called 'nb'.
        """
        if self.ctype == 'eraLDBODY':
            return 'nb, _' + self.name
        return ("_" if self.is_ptr else "*_") + self.name


class StatusCode(Variable):
    def __init__(self, ctype: str, doc: FunctionDoc, funcname: str) -> None:
        self.name = "c_retval"
        self.inout_state = "stat"
        self.ctype = "int"
        self.shape = ()

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

    def to_python(self) -> str:
        return "\n".join(
            ["{", *[f"    {k!r}: {v!r}," for k, v in self._statuscodes.items()], "}"]
        )


class Return(Variable):

    def __init__(self, ctype, doc):
        self.name = 'c_retval'
        self.inout_state = "ret"
        self.ctype = ctype
        self.shape = ()


class Function:
    """
    A class representing a C function.

    Parameters
    ----------
    name : str
        The name of the function
    source_path : pathlib.Path
        Directory with the file containing the function implementation.
    """

    def __init__(self, name, source_path):
        self.name = name
        self.pyname = name.split('era')[-1].lower()

        pattern = fr"\n([^\n]+{name} ?\([^)]+\)).+?(/\*.+?\*/)"
        p = re.compile(pattern, flags=re.DOTALL | re.MULTILINE)
        search = p.search((source_path / (self.pyname + ".c")).read_text())
        self.cfunc = " ".join(search.group(1).split())
        self.doc = FunctionDoc(search.group(2))

        self.args = []
        for arg in re.search(r"\(([^)]+)\)", self.cfunc).group(1).split(', '):
            self.args.append(Argument(arg, self.doc))
        self.ret = re.search(f"^(.*){name}", self.cfunc).group(1).strip()
        if self.ret == "int" and self.name not in ("eraTpors", "eraTporv"):
            self.args.append(StatusCode(self.ret, self.doc, name))
        elif self.ret != "void":
            self.args.append(Return(self.ret, self.doc))

    def args_by_inout(self, inout_filter):
        """
        Gives all of the arguments and/or returned values, depending on whether
        they are inputs, outputs, etc.

        The value for `inout_filter` should be a string containing anything
        that arguments' `inout_state` attribute produces.  Currently, that can be:

          * "in" : input
          * "out" : output
          * "inout" : something that's could be input or output (e.g. a struct)
          * "ret" : the return value of the C function
          * "stat" : the return value of the C function if it is a status code

        It can also be a "|"-separated string giving inout states to OR
        together.
        """
        return [arg for arg in self.args if arg.inout_state in inout_filter.split("|")]

    @property
    def user_dtype(self):
        """The non-standard dtype, if any, needed by this function's ufunc.

        This would be any structured array for any input or output, but
        we give preference to LDBODY, since that also decides that the ufunc
        should be a generalized ufunc.
        """
        user_dtype = None
        for arg in self.args_by_inout('in|inout|out'):
            if arg.ctype == 'eraLDBODY':
                return arg.dtype
            if user_dtype is None and arg.dtype not in ("dt_double", "dt_int"):
                user_dtype = arg.dtype

        return user_dtype

    @property
    def signature(self):
        """Possible signature, if this function should be a gufunc."""
        if all(arg.signature_shape == '()'
               for arg in self.args_by_inout('in|inout|out')):
            return None

        return '->'.join(
            [','.join([arg.signature_shape for arg in args])
             for args in (self.args_by_inout('in|inout'),
                          self.args_by_inout('inout|out|ret|stat'))])

    @property
    def python_call(self):
        result = _assemble_py_func_call(
            "ufunc." + self.pyname,
            in_args=[arg.name for arg in self.args_by_inout("in|inout")],
            out_args=[arg.name for arg in self.args_by_inout("inout|out|stat|ret")],
        )
        if len(result) < 75:
            return result

        if result.index('(') < 75:
            return result.replace('(', '(\n        ')

        split_point = result[:75].rfind(',') + 1
        return ('(' + result[:split_point] + '\n    '
                + result[split_point:].replace(' =', ') ='))


class Constant:

    def __init__(self, name, value, doc):
        self.name = name.replace("ERFA_", "")
        self.value = value.replace("ERFA_", "")
        self.doc = doc


class TestFunction:
    """Function holding information about a test in t_erfa_c.c"""
    def __init__(self, name, t_erfa_c, nin, ninout, nout):
        self.name = name
        # Get lines that test the given erfa function: capture everything
        # between a line starting with '{' after the test function definition
        # and the first line starting with '}' or ' }'.
        pattern = fr"\nstatic void t_{name}\(" + r".+?(^\{.+?^\s?\})"
        search = re.search(pattern, t_erfa_c, flags=re.DOTALL | re.MULTILINE)
        self.lines = search.group(1).split('\n')
        # Number of input, inplace, and output arguments.
        self.nin = nin
        self.ninout = ninout
        self.nout = nout
        # Dict of dtypes for variables, filled by define_arrays().
        self.var_dtypes = {}

    @classmethod
    def from_function(cls, func, t_erfa_c):
        """Initialize from a function definition."""
        return cls(name=func.pyname, t_erfa_c=t_erfa_c,
                   nin=len(func.args_by_inout('in')),
                   ninout=len(func.args_by_inout('inout')),
                   nout=len(func.args_by_inout('out')))

    def xfail(self):
        """Whether the python test produced for this function will fail when the xfail condition is verified.

        Right now this will be true for functions without inputs such
        as eraIr with numpy < 1.24.
        """
        return (
            "np.__version__ < '1.24', reason='numpy < 1.24 do not support no-input ufuncs'"
            if self.nin + self.ninout == 0 and self.name != "zpv"
            else None
        )

    def pre_process_lines(self):
        """Basic pre-processing.

        Combine multi-part lines, strip braces, semi-colons, empty lines.
        """
        lines = []
        line = ''
        for part in self.lines:
            part = part.strip()
            if part in ('', '{', '}'):
                continue
            line += part + ' '
            if part.endswith(';'):
                lines.append(line.strip()[:-1])
                line = ''
        return lines

    def define_arrays(self, line):
        """Check variable definition line for items also needed in python.

        E.g., creating an empty astrom structured array.
        """
        defines = []
        # Split line in type and variables.
        # E.g., "double x, y, z" will give ctype='double; variables='x, y, z'
        ctype, _, variables = line.partition(' ')
        for var in variables.split(','):
            var = var.strip()
            # Is variable an array?
            name, _, rest = var.partition('[')
            # If not, or one of iymdf or ihmsf, ignore (latter are outputs only).
            if not rest or rest[:2] == '4]':
                continue
            if ctype == 'eraLDBODY':
                # Special case, since this should be recarray for access similar
                # to C struct.
                v_dtype = 'dt_eraLDBODY'
                v_shape = rest[:rest.index(']')]
                extra = ".view(np.recarray)"
            else:
                # Temporarily create an Argument, so we can use its attributes.
                # This translates, e.g., double pv[2][3] to dtype dt_pv.
                v = Argument(ctype + ' ' + var.strip(), '')
                v_dtype = v.dtype
                v_shape = v.shape if v.signature_shape != '()' else '()'
                extra = ""
            self.var_dtypes[name] = v_dtype
            v_dtype = "float" if v_dtype == "dt_double" else "erfa_ufunc." + v_dtype
            defines.append(f"{name} = np.empty({v_shape}, {v_dtype}){extra}")

        return defines

    def to_python(self):
        """Lines defining the body of a python version of the test function."""
        # TODO: this is quite hacky right now!  Would be good to let function
        # calls be understood by the Function class.

        # Name of the erfa C function, so that we can recognize it.
        era_name = 'era' + self.name.capitalize()
        # Collect actual code lines, without ";", braces, etc.
        lines = self.pre_process_lines()
        out = []
        for line in lines:
            # In ldn ufunc, the number of bodies is inferred from the array size,
            # so no need to keep the definition.
            if line == 'n = 3' and self.name == 'ldn':
                continue

            # Are we dealing with a variable definition that also sets it?
            # (hack: only happens for double).
            if line.startswith('double') and '=' in line:
                # Complete hack for single occurrence.
                if line.startswith('double xyz[] = {'):
                    out.append(f"xyz = np.array([{line[16:-1]}])")
                else:
                    # Put each definition on a separate line.
                    out.extend([part.strip() for part in line[7:].split(',')])
                continue

            # Variable definitions: add empty array definition as needed.
            if line.startswith(('double', 'int', 'char', 'eraASTROM', 'eraLDBODY')):
                out.extend(self.define_arrays(line))
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

            # Call of test function vvi or vvd.
            if line.startswith('v'):
                line = line.replace(era_name, self.name)
                # Can call simple functions directly.  Those need little modification.
                if self.name + '(' in line:
                    line = line.replace(self.name + '(', f"erfa_ufunc.{self.name}(")

            # Call of function that is being tested.
            elif era_name in line:
                # correct for LDBODY (complete hack!)
                line = line.replace('3, b', 'b').replace('n, b', 'b')
                name, arguments = _get_funcname_and_args(
                    line, era_name, f"erfa_ufunc.{self.name}"
                )
                # Remove leading zeros from numbers since python cannot deal with them.
                args = [
                    (arg.lstrip("0") or "0") if arg.isdigit() else arg
                    for arg in arguments
                ]
                # Get input and output arguments.
                in_args = [arg.replace('&', '') for arg in args[:self.nin+self.ninout]]
                out_args = ([arg.replace('&', '') for arg in args[-self.nout-self.ninout:]]
                            if len(args) > self.nin else [])
                # If the call assigned something, that will have been the status.
                # Prepend any arguments assigned in the call.
                if " = " in name:
                    status, name = name.split(" = ", 1)
                    out_args.append(status)
                line = _assemble_py_func_call(name, in_args, out_args)
                if 'astrom' in out_args:
                    out.append(line)
                    line = 'astrom = astrom.view(np.recarray)'

            # In some test functions, there are calls to other ERFA functions.
            # Deal with those in a super hacky way for now.
            elif line.startswith('eraA'):
                name, args = _get_funcname_and_args(line, "eraA", "erfa_ufunc.a")
                line = _assemble_py_func_call(
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
                line = _assemble_py_func_call(name, in_args=args[:2], out_args=args[2:])

            # Input number setting.
            elif '=' in line:
                # Small clean-up.
                line = line.replace('=  ', '= ')
                # Hack to make astrom element assignment work.
                if line.startswith('astrom'):
                    out.append('astrom = np.zeros((), erfa_ufunc.dt_eraASTROM).view(np.recarray)')
                # Change access to p and v elements for double[2][3] pv arrays
                # that were not caught by the general replacement above (e.g.,
                # with names not equal to 'pv')
                name, _, rest = line.partition('[')
                if (rest and rest[0] in '01' and name in self.var_dtypes
                        and self.var_dtypes[name] == 'dt_pv'):
                    line = name + "[" + ("'p'" if rest[0] == "0" else "'v'") + rest[1:]

            out.append(line)

        return out


def _get_funcname_and_args(
    line: str, c_prefix: str, py_prefix: str
) -> tuple[str, list[str]]:
    funcname, args = line.replace(c_prefix, py_prefix).split("(", 1)
    return funcname, [arg.strip() for arg in args.removesuffix(")").split(",")]


def _assemble_py_func_call(name: str, in_args: list[str], out_args: list[str]) -> str:
    return f"{', '.join(out_args)} = {name}({', '.join(in_args)})"


def main(srcdir: Path, templateloc: Path) -> None:
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(templateloc))
    env.filters["surround"] = lambda elems, pre, post: [pre + e + post for e in elems]

    funcs = [
        Function(name, srcdir)
        for name in re.findall(
            r"\w+ (\w+)\(.*?\);", (srcdir / "erfa.h").read_text(), flags=re.DOTALL
        )
    ]

    constants = []
    for chunk in (srcdir / "erfam.h").read_text().split("\n\n"):
        doc = re.findall(r"/\* (.+?) \*/\n", chunk, flags=re.DOTALL)
        constants.extend(
            Constant(name, value, doc)
            for name, value in re.findall(
                r"#define (ERFA_\w+?) (.+?)$", chunk, flags=re.DOTALL | re.MULTILINE
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
        TestFunction.from_function, t_erfa_c=(srcdir / "t_erfa_c.c").read_text()
    )
    (testloc / testfn).write_text(
        Environment(loader=FileSystemLoader(testloc))
        .get_template(testfn + ".templ")
        .render(test_funcs=sorted(map(create_test_funcs, funcs), key=lambda f: f.name))
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
