#!/usr/bin/env python3
import ast
import csv
import os


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.List):
        return [_eval_node(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)
    if isinstance(node, ast.NameConstant):
        return node.value
    raise ValueError(f"unsupported default expression: {ast.dump(node)}")


def _load_defaults(path):
    with open(path, "r") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "defaults":
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        if node.value.func.id == "dict":
                            defaults = {}
                            for kw in node.value.keywords:
                                if kw.arg is None:
                                    continue
                                defaults[kw.arg] = _eval_node(kw.value)
                            return defaults
    raise SystemExit("defaults dict not found in utils/settings.py")


def _fmt_value(val):
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def _tex_escape(s):
    return s.replace("_", "\\_")


def _split_key(key, max_seg=12):
    parts = key.split("_")
    lines = []
    line = []
    for part in parts:
        if sum(len(p) for p in line) + len(line) + len(part) > max_seg:
            if line:
                lines.append("_".join(line))
            line = [part]
        else:
            line.append(part)
    if line:
        lines.append("_".join(line))
    return "\\\\".join(lines)


def main():
    defaults = _load_defaults(os.path.join("utils", "settings.py"))
    rows = sorted(defaults.items(), key=lambda x: x[0])

    out_csv = os.path.join("docs", "figures", "config_table.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "default"])
        for key, val in rows:
            writer.writerow([key, _fmt_value(val)])

    out_tex = os.path.join("docs", "figures", "config_table.tex")
    with open(out_tex, "w") as f:
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{longtable}{p{0.32\\linewidth} p{0.15\\linewidth} p{0.45\\linewidth}}\n")
        f.write("\\caption{Full configuration parameter defaults (from utils/settings.py).}\\\\label{tab:config-full}\\\\\n")
        f.write("\\hline\n")
        f.write("Key & Default & Notes \\\\\n")
        f.write("\\hline\n")
        f.write("\\endfirsthead\n")
        f.write("\\hline\n")
        f.write("Key & Default & Notes \\\\\n")
        f.write("\\hline\n")
        f.write("\\endhead\n")
        for key, val in rows:
            key_tex = _tex_escape(_split_key(key))
            val_tex = _tex_escape(_fmt_value(val))
            f.write(f"{key_tex} & {val_tex} & \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{longtable}\n")
        f.write("\\normalsize\n")
        f.write("\\setlength{\\tabcolsep}{6pt}\n")

    print(f"[config-table] wrote {out_csv} and {out_tex}")


if __name__ == "__main__":
    main()
