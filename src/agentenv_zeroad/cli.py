"""`agent-env zeroad`: build the engine and env images and register the env, and remove leftover engine containers.

`setup` builds `zeroad-engine:0.28` (scripts/build-engine.sh, cached) unless it exists, builds the env image on top of
it (the repo's Dockerfile, with the agentenv-game-env archive pyproject.toml pins), stores it and registers it as the
MCP server env `zeroad` on the `server` provider, which the bundle's tasks deploy. `setup --agents` also builds and
registers the two brains as A2A agents (`zeroad-commander`, `zeroad-luna`, from agents/), which the model tasks deploy.
`setup --live` builds the stream side instead: the patched engine (`scripts/build-engine.sh rt`, zeroad-engine:0.28-rt),
the env image on it, the live image on that (streamer/Dockerfile: Chrome and the director mod), and registers it as
the env `zeroad-live`, which the `livestream` tasks deploy.
"""

from __future__ import annotations

import asyncio
import subprocess
import tomllib
from pathlib import Path

import click

from agentenv_zeroad.engine import DEFAULT_IMAGE, reap_orphans

ENVIRONMENT_NAME = "zeroad"
LIVE_ENV = "zeroad-live"
RT_IMAGE = "zeroad-engine:0.28-rt"
AGENTS = {  # A2A agent id: its directory under agents/, what it is
    "zeroad-commander": ("zeroad-commander", "big-model strategist"),
    "zeroad-luna": ("zeroad-luna", "GPT-6 Luna player"),
}
REPO_HINT = "a checkout of agentenv-0ad-plugin (with Dockerfile, engine/ and scripts/build-engine.sh)"


@click.group()
def zeroad():
    """0 A.D. Live Arena: build and register the env, and clean up engine containers."""


def checkout(source: Path | None) -> Path:
    """The checkout to build from: `source`, else the one an editable install runs from, else the cwd."""
    for root in [source] if source else [Path(__file__).resolve().parents[2], Path.cwd()]:
        if root and (root / "Dockerfile").is_file() and (root / "scripts/build-engine.sh").is_file():
            return root
    raise click.UsageError(f"no checkout found; pass --source with {REPO_HINT}")


def game_env(root: Path) -> str:
    """The agentenv-game-env archive pyproject.toml pins, for the env image."""
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    pinned = next((d for d in deps if d.startswith("agentenv-game-env @ ")), None)
    if pinned is None:
        raise click.ClickException(f"{root}/pyproject.toml pins no agentenv-game-env archive")
    return pinned.removeprefix("agentenv-game-env @ ").strip()


def image_exists(image: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    except OSError as e:
        raise click.ClickException(f"docker is not available: {e}") from e


def live_build_commands(root: Path, image: str) -> list[list[str]]:
    """The docker builds `setup --live` runs: the rt engine when missing, the env image on it, the live image on top."""
    cmds = []
    if not image_exists(RT_IMAGE):
        cmds.append([str(root / "scripts/build-engine.sh"), "rt"])
    env_image = f"{image}-rt-base"
    cmds += build_commands(root, env_image, RT_IMAGE)
    cmds.append(
        ["docker", "build", "-f", str(root / "streamer/Dockerfile"), "--build-arg", f"ENV_IMAGE={env_image}"]
        + ["-t", image, str(root)]
    )
    return cmds


def build_commands(root: Path, image: str, engine_image: str) -> list[list[str]]:
    """The docker builds `setup` runs: the engine image when missing, then the env image on top of it."""
    cmds = []
    if engine_image == DEFAULT_IMAGE and not image_exists(engine_image):
        cmds.append([str(root / "scripts/build-engine.sh")])
    cmds.append(
        [
            "docker",
            "build",
            "--build-arg",
            f"GAME_ENV={game_env(root)}",
            "--build-arg",
            f"ENGINE_IMAGE={engine_image}",
            "-t",
            image,
            str(root),
        ]
    )
    return cmds


@zeroad.command()
@click.option("--id", "env_id", default=ENVIRONMENT_NAME, show_default=True, help="Env id to register.")
@click.option("--source", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Checkout to build.")
@click.option("--engine-image", default=DEFAULT_IMAGE, show_default=True, help="The engine image to build on.")
@click.option("--image", help="Register this existing local env image instead of building one.")
@click.option("--agents", is_flag=True, help="Also build and register the A2A agents zeroad-commander and zeroad-luna.")
@click.option("--agents-only", is_flag=True, help="Only build and register the A2A agents.")
@click.option("--live", is_flag=True, help="Build the stream env (rt engine, Chrome, director mod) as zeroad-live.")
def setup(
    env_id: str, source: Path | None, engine_image: str, image: str | None, agents: bool, agents_only: bool, live: bool
):
    """Build the engine and env images and register the env as an MCP server env on the `server` provider."""
    from agent_env.artifact import DockerImageArtifact
    from agent_env.env import MCPServerEnv

    if live and env_id == ENVIRONMENT_NAME:
        env_id = LIVE_ENV
    if agents or agents_only:
        root = checkout(source)
        for agent_id in AGENTS:
            register_agent(root, agent_id)
    if agents_only:
        return
    if image is None:
        root = checkout(source)
        image = f"mcp-server-{env_id}"
        cmds = live_build_commands(root, image) if live else build_commands(root, image, engine_image)
        for cmd in cmds:
            click.echo("$ " + " ".join(cmd))
            if subprocess.run(cmd).returncode:
                raise click.ClickException(f"{cmd[0]} failed")
    click.echo("Storing the image (docker save, can take a few minutes for the engine's 2 GB)")
    artifact = DockerImageArtifact.put(
        id=f"mcp-server-{env_id}", image_name=image, description="0 A.D. env: the engine and its MCP server"
    )
    env = MCPServerEnv.put(
        id=env_id, docker_image_artifact=artifact, environment_name=ENVIRONMENT_NAME, env_provider_type="server"
    )
    click.echo(f"Registered env {env.id!r} version {env.version} (image {artifact.image_name})")
    click.echo("Next: agent-env run zeroad --task " + ("livestream-petra" if live else "smoke"))


def agent_build_command(root: Path, agent_id: str) -> list[str]:
    path, _ = AGENTS[agent_id]
    return [
        "docker",
        "build",
        "-t",
        f"a2a-agent-{agent_id}",
        "-f",
        str(root / "agents" / path / "Dockerfile"),
        str(root),
    ]


def register_agent(root: Path, agent_id: str) -> None:
    """Build agents/<agent> (with the package's brains) from the repo root and register it as `agent_id`."""
    from agent_env.a2a_agent import A2AAgent
    from agent_env.artifact import DockerImageArtifact

    cmd = agent_build_command(root, agent_id)
    click.echo("$ " + " ".join(cmd))
    if subprocess.run(cmd).returncode:
        raise click.ClickException(f"docker build of {agent_id} failed")
    image, what = cmd[3], AGENTS[agent_id][1]
    artifact = DockerImageArtifact.put(
        id=image,
        image_name=image,
        description=f"0 A.D. {what} agent",
        build_context_path=str(root),
        dockerfile_path=cmd[5],
    )
    agent = A2AAgent.put(id=agent_id, docker_image_artifact=artifact)
    click.echo(f"Registered A2A agent {agent.id!r} version {agent.version} ({what})")


@zeroad.command()
def reap():
    """Remove engine containers (zeroad-engine-*) whose owner process is gone."""
    removed = asyncio.run(reap_orphans())
    click.echo(f"removed {len(removed)}: {', '.join(removed)}" if removed else "no orphaned engine containers")
