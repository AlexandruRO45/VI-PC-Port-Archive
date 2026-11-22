"""Expand single-block pure value functions for the semantic interpreters' single-function execution model."""
import re


_FUNCTION = re.compile(r"^(%\S+) = OpFunction (%\S+) (?:Pure|Const) %\S+$", re.MULTILINE)
_ID = re.compile(r"%[\w.]+")


def inline_pure_functions(text: str) -> str:
    if "OpFunctionCall" not in text or not _FUNCTION.search(text):
        return text
    functions = {}
    entry = []
    lines = iter(text.splitlines())
    for line in lines:
        match = _FUNCTION.fullmatch(line.strip())
        if not match:
            entry.append(line)
            continue
        original = [line]
        for line in lines:
            original.append(line)
            if line.strip() == "OpFunctionEnd":
                break
        params, body, returned = [], [], None
        labels = 0
        for line in original[1:-1]:
            words = line.split()
            if len(words) == 4 and words[2] == "OpFunctionParameter":
                params.append(words[0])
            elif len(words) == 3 and words[2] == "OpLabel":
                labels += 1
            elif len(words) == 2 and words[0] == "OpReturnValue":
                returned = words[1]
            elif len(words) >= 4 and words[1] == "=" and words[2] not in ("OpVariable", "OpFunctionCall", "OpPhi"):
                body.append((words[0], line))
            else:
                break
        else:
            if labels == 1 and returned is not None:
                functions[match[1]] = (params, body, returned)
                continue
        entry.extend(original)   # let the interpreter report unsupported control flow

    prefix = "%inline_"
    while prefix in text:
        prefix += "_"
    expanded = []
    for line in entry:
        words = line.split()
        if len(words) >= 5 and words[2] == "OpFunctionCall" and words[4] in functions:
            params, body, returned = functions[words[4]]
            if len(params) != len(words[5:]):
                raise ValueError(f"wrong argument count in {line.strip()}")
            names = dict(zip(params, words[5:]))
            for local, instruction in body:
                names[local] = f"{prefix}{len(expanded)}"
                expanded.append(_ID.sub(lambda m: names.get(m[0], m[0]), instruction))
            expanded.append(f"{words[0]} = OpCopyObject {words[3]} {names.get(returned, returned)}")
        else:
            expanded.append(line)
    return "\n".join(expanded)
