import os
import posixpath
import re

GITHUB_BASE = "https://github.com/square-key-labs/strawgo-ai"
BRANCH = "main"
EXCLUDED_DIRS = ("agents",)

FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})", re.MULTILINE)
INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)+?\1", re.DOTALL)
LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _split_fenced(markdown):
    parts = []
    lines = markdown.splitlines(keepends=True)
    buf = []
    in_code = None
    for line in lines:
        m = FENCE.match(line)
        if in_code is None:
            if m:
                parts.append((False, "".join(buf)))
                buf = [line]
                in_code = m.group(2)[0] * len(m.group(2))
            else:
                buf.append(line)
        else:
            buf.append(line)
            if m and m.group(2).startswith(in_code) and line.strip() == m.group(2):
                parts.append((True, "".join(buf)))
                buf = []
                in_code = None
    parts.append((in_code is not None, "".join(buf)))
    return parts


def _github_url(repo_root, abs_path, anchor):
    rel = os.path.relpath(abs_path, repo_root).replace(os.sep, "/")
    kind = "tree" if os.path.isdir(abs_path) else "blob"
    url = f"{GITHUB_BASE}/{kind}/{BRANCH}/{rel}"
    return url + ("#" + anchor if anchor else "")


def _rewrite_target(target, page_src, docs_dir, repo_root):
    if target.startswith(("#", "/")) or SCHEME.match(target):
        return target
    path, _, anchor = target.partition("#")
    if not path:
        return target
    rel_in_docs = posixpath.normpath(posixpath.join(posixpath.dirname(page_src), path))
    abs_path = os.path.normpath(os.path.join(docs_dir, rel_in_docs))
    outside = rel_in_docs.startswith("../")
    excluded = rel_in_docs.split("/")[0] in EXCLUDED_DIRS
    is_page = os.path.isfile(abs_path) and abs_path.endswith(".md")
    if not os.path.exists(abs_path):
        return target
    if outside or excluded or not is_page:
        return _github_url(repo_root, abs_path, anchor)
    return target


def _rewrite_text(text, page_src, docs_dir, repo_root):
    def link(m):
        return m.group(1) + _rewrite_target(m.group(2), page_src, docs_dir, repo_root) + m.group(3)

    out = []
    last = 0
    for code in INLINE_CODE.finditer(text):
        out.append(LINK.sub(link, text[last:code.start()]))
        out.append(code.group(0))
        last = code.end()
    out.append(LINK.sub(link, text[last:]))
    return "".join(out)


def on_page_markdown(markdown, page, config, files):
    docs_dir = config["docs_dir"]
    repo_root = os.path.dirname(os.path.abspath(config["config_file_path"]))
    page_src = page.file.src_uri
    return "".join(
        chunk if is_code else _rewrite_text(chunk, page_src, docs_dir, repo_root)
        for is_code, chunk in _split_fenced(markdown)
    )
