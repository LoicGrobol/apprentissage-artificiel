import functools
import os.path
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import click
import jinja2 as j2
import pathspec
import tomllib
import yaml
from jinja2.loaders import FileSystemLoader
from livereload import Server
from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin


# TODO: uuuuuuglyyyyy
def render_md(
    source_file: pathlib.Path,
    environment: j2.Environment,
    source_root: pathlib.Path,
    context: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    if context is None:
        context = {}

    layout = "default"

    inner_template = environment.get_template(str(source_file.relative_to(source_root)))
    markdown_content = inner_template.render(context)
    md = (
        MarkdownIt("commonmark", renderer_cls=RendererHTML)
        .use(front_matter_plugin)
        .use(footnote_plugin)
        .enable("table")
    )
    tokens = md.parse(markdown_content)
    if tokens[0].type == "front_matter":
        front_matter = yaml.safe_load(tokens[0].content)
        context["page"] = front_matter
        layout = front_matter.get("layout", layout)
        permalink = front_matter.get("permalink")
    else:
        permalink = None

    outer_template = environment.get_template(f"templates/{layout}.html.jinja")
    content = md.render(markdown_content)
    return outer_template.render({**context, "content": content}), permalink


def build_site(
    build_dir: pathlib.Path,
    source_dir: pathlib.Path,
    config_overrides: dict[str, Any] | None = None,
):
    with (source_dir / "config.toml").open("rb") as in_stream:
        config = tomllib.load(in_stream)

    if config_overrides is not None:
        config.update(config_overrides)

    print(f"{source_dir / 'assets'} → {build_dir / 'assets'}", file=sys.stderr)
    shutil.copytree(source_dir / "assets", build_dir / "assets", dirs_exist_ok=True)

    env = j2.Environment(
        loader=FileSystemLoader(source_dir),
        autoescape=j2.select_autoescape(),
        auto_reload=False,
    )
    env.globals["site"] = config

    # Horrible hack to auto-import macros
    env.globals.update(
        {
            k: v
            for k, v in env.get_template("templates/macros.jinja").module.__dict__.items()
            if not k.startswith("_")
        }
    )

    contents_dir = source_dir / "contents"
    for f in contents_dir.glob("**/*.md.jinja"):
        print(f"Processing {f}", file=sys.stderr)
        content, permalink = render_md(
            source_file=f,
            source_root=source_dir,
            environment=env,
        )
        if permalink is None:
            target = build_dir / f.relative_to(contents_dir).with_suffix("html")
        else:
            if permalink.endswith("/"):
                permalink = f"{permalink}index.html"
            if permalink.startswith("/"):
                target = build_dir / permalink[1:]
            else:
                target = build_dir / f.parent.relative_to(contents_dir) / permalink
        print(f"{f} → {target}", file=sys.stderr)
        target.write_text(content)


def build_jl(build_dir: pathlib.Path, jupyterlite_dir: pathlib.Path, notebooks_dir: pathlib.Path):
    notebooks_target_dir = build_dir / "notebooks"
    notebooks_target_dir.mkdir(exist_ok=True)

    for d in notebooks_dir.glob("*/"):
        shutil.copytree(d, notebooks_target_dir / d.name, dirs_exist_ok=True)

    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "jupytext",
            "--to",
            "ipynb",
            *notebooks_target_dir.glob("**/*.py.md"),
        ]
    )
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "jupyter",
            "lite",
            "build",
            "--contents",
            str(build_dir / "notebooks"),
            "--lite-dir",
            jupyterlite_dir,
            "--output-dir",
            str(build_dir / "jupyterlite"),
        ],
        cwd=build_dir,
    )
    # JupyterLite stop polluting my build dir >:[
    (build_dir / ".jupyterlite.doit.db").unlink()


def _build(
    build_dir: pathlib.Path,
    source_dir: pathlib.Path,
    config_overrides: dict[str, Any] | None = None,
):
    build_dir = build_dir.resolve()
    build_dir.mkdir(exist_ok=True, parents=True)
    source_dir = source_dir.resolve()

    build_site(
        build_dir=build_dir,
        config_overrides=config_overrides,
        source_dir=source_dir / "site",
    )
    build_jl(
        build_dir=build_dir,
        jupyterlite_dir=source_dir / "jupyterlite",
        notebooks_dir=source_dir / "notebooks",
    )


def get_all_gitignores(base_dir: pathlib.Path) -> tuple[list[str], list[pathlib.Path]]:
    # First try to find a .git
    root = base_dir.resolve()
    while not (root / ".git").exists():
        # Filesystem root: abort mission
        if root.parent == root:
            root = base_dir
            break
        root = root.parent
    gitignores = list(root.glob("**/*.gitignore"))
    patterns = []
    for f in gitignores:
        prefix = str(f.parent.relative_to(root))
        with f.open() as in_stream:
            for p in in_stream:
                if p and not p.isspace() and not p.startswith("#"):
                    patterns.append(os.path.join(prefix, p.strip()))  # noqa: PTH118
    return patterns, gitignores


@click.group()
def cli():
    pass


@cli.command(help="Build the site")
@click.argument(
    "source_dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--build-dir",
    default=pathlib.Path.cwd() / "_build",
    show_default=True,
    type=click.Path(writable=True, file_okay=False, path_type=pathlib.Path),
)
def build(build_dir: pathlib.Path, source_dir: pathlib.Path):
    _build(build_dir=build_dir, source_dir=source_dir)


@cli.command(help="Watch and serve the site")
@click.argument(
    "source_dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--build-dir",
    default=pathlib.Path.cwd() / "_build",
    show_default=True,
    type=click.Path(writable=True, file_okay=False, path_type=pathlib.Path),
)
@click.option("--port", default=5000, show_default=True, type=int)
@click.option(
    "--watch / --no-watch",
    default=False,
)
def serve(build_dir: pathlib.Path, port: int, source_dir: pathlib.Path, watch: bool):
    cmd = functools.partial(
        _build,
        build_dir=build_dir,
        source_dir=source_dir,
        config_overrides={"domain": f"http://localhost:{port}", "baseurl": ""},
    )
    cmd()
    server = Server()
    if watch:
        # TODO: doesn't seem to work because liverload is passing full paths and pathspec is expecting
        # paths relative to the repo root
        ignore_patterns, gitignores = get_all_gitignores(source_dir)
        if ignore_patterns:
            ps = pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)
            # TODO: update these when gitignores themselves change?

            ignore = ps.match_file
        else:
            ignore = None
        server.watch(source_dir, cmd, delay=2, ignore=ignore)
    server.serve(host="localhost", port=port, root=build_dir)


@cli.command(help="Cleanup build files etc.")
@click.argument(
    "source_dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--build-dir",
    default=pathlib.Path.cwd() / "_build",
    show_default=True,
    type=click.Path(writable=True, file_okay=False, path_type=pathlib.Path),
)
def clean(build_dir: pathlib.Path, source_dir: pathlib.Path):
    for p in [build_dir]:
        if p == source_dir:
            raise ValueError(f"Attempting to clean source dir {source_dir}")
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        else:
            print(f"File not found for cleanup: {p}", file=sys.stderr)


if __name__ == "__main__":
    cli()
