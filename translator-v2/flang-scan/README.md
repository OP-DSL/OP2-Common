# op2-flang-scan

A small C++ tool that reads a preprocessed free-form Fortran source file, walks Flang's parse tree, and emits a JSON document. Each document has a `path` and an ordered `events` array. If Flang cannot parse the unit, the object has an `error` string and empty `events` instead.

Events collected:

- **`op_par_loop_N` / `op_decl_const`** — each matching `CALL` site, with source `location` and a tagged argument tree (`name`, folded `int`, `string`, nested `call`, or `raw` source).
- **`subroutine_subprogram` / `function_subprogram`** — every subroutine or function (plus nested calls): dummy `parameters`; `depends` (callee names from `CALL` and parenthesised `funcref`s); cooked `source` text of the whole subprogram; `locals` (locally declared arrays, with every identifier used in a shape bound); a flattened `assignments` / `calls` walk of the execution part; typed `decls`; and a nested `stmts` tree. Function events also carry `result_name` and `result_type` (from `RESULT(...)` and a prefix type such as `real function foo`, or `null` if omitted).

The Python side invokes this binary and consumes the JSON when the translator is run with `--parser flang`.

## JSON format

```jsonc
{
  "path": "airfoil_op.F90",
  "events": [
    {
      "kind": "op_par_loop_5",
      "location": {"line": 120, "column": 3},
      "args": [
        {"kind": "name",   "value": "save_soln"},
        {"kind": "name",   "value": "edges"},
        {"kind": "call",   "name": "op_arg_dat",
         "args": [
           {"kind": "name",   "value": "p_q"},
           {"kind": "int",    "value": -1},
           {"kind": "name",   "value": "op_id"},
           {"kind": "int",    "value": 4},
           {"kind": "string", "value": "real(8)"},
           {"kind": "name",   "value": "op_read"}
         ]}
      ]
    },
    {
      "kind": "op_decl_const",
      "location": {"line": 42, "column": 3},
      "args": [
        {"kind": "name",   "value": "gam"},
        {"kind": "int",    "value": 1},
        {"kind": "string", "value": "real(8)"}
      ]
    },
    {
      "kind": "subroutine_subprogram",
      "name": "res_calc",
      "location": {"line": 60, "column": 15},
      "parameters": ["x1", "x2", "q1", "q2", "adt1", "adt2", "res1", "res2"],
      "depends": [],
      "source": "subroutine res_calc(...)\n  ...\nend subroutine\n",
      "locals": [
        {"name": "q1", "dims": []}
      ],
      "assignments": [
        {
          "line": 82,
          "lhs": {"kind": "funcref", "name": "res1", "args": [{"kind": "int_lit", "text": "1", "kind_text": null}]},
          "rhs": {
            "kind": "binary", "op": "+",
            "left":  {"kind": "funcref", "name": "res1", "args": [{"kind": "int_lit", "text": "1", "kind_text": null}]},
            "right": {"kind": "name", "value": "f"}
          }
        }
      ],
      "calls": [],
      "decls": [
        {
          "kind": "type_decl",
          "type": {"kind": "intrinsic", "base": "real", "kind_text": "8", "charlen": null},
          "is_parameter": false,
          "dim": null,
          "entities": [
            {"name": "dx", "dim": null, "init": null},
            {
              "name": "q1",
              "dim": {
                "kind": "explicit",
                "shape": [{"lb": null, "ub": {"kind": "int_lit", "text": "4", "kind_text": null}}]
              },
              "init": null
            }
          ]
        }
      ],
      "stmts": [
        {
          "kind": "assign",
          "line": 82,
          "lhs": {"kind": "funcref", "name": "res1", "args": [{"kind": "int_lit", "text": "1", "kind_text": null}]},
          "rhs": {
            "kind": "binary", "op": "+",
            "left":  {"kind": "funcref", "name": "res1", "args": [{"kind": "int_lit", "text": "1", "kind_text": null}]},
            "right": {"kind": "name", "value": "f"}
          }
        }
      ]
    }
    // function_subprogram events match subroutine_subprogram, plus:
    //   "result_name": str|null,
    //   "result_type": <type>|null
  ]
}
```

`decls` may also contain `parameter_stmt` (`defs: [{name, value}]`) and `data_stmt` (`sets` of objects/values), and `stmts` preserves control flow. Body expressions use `name`, `part_ref`, `funcref`, `triplet`, `binary`, `unary`, `paren`, `int_lit`/`real_lit`/`logical_lit`/`char_lit`, `raw`, or `unsupported`.
