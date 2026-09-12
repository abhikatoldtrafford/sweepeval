# sweepeval as a container, for CI runners that would rather not install
# Python toolchains.
#
#   docker run --rm ghcr.io/abhikatoldtrafford/sweepeval demo
#   docker run --rm -v "$PWD/.sweepeval:/work/.sweepeval" \
#     ghcr.io/abhikatoldtrafford/sweepeval \
#     sweep https://your-endpoint --key "$KEY" --yes
#
# Two properties worth stating, because both are easy to lose in a Dockerfile:
#
# * It runs as a non-root user. The tool writes artifacts to a working
#   directory the caller mounts, and a container that writes them as root
#   leaves a directory the caller cannot delete.
# * The entrypoint is the CLI, not a shell. `docker run <image> demo` is the
#   whole invocation.

FROM python:3.12-slim AS build

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

FROM python:3.12-slim

LABEL org.opencontainers.image.title="sweepeval" \
      org.opencontainers.image.description="Zero-config, black-box sweep and benchmark engine for LLM and agent systems." \
      org.opencontainers.image.source="https://github.com/abhikatoldtrafford/sweepeval" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# A writable working directory owned by the runtime user, so a bind-mounted
# .sweepeval/ does not come back root-owned.
RUN useradd --create-home --uid 10001 sweepeval
WORKDIR /work
RUN chown sweepeval:sweepeval /work
USER sweepeval

ENTRYPOINT ["sweepeval"]
CMD ["--help"]
