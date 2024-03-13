"""Define the workflow class"""

import contextlib
import inspect
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import git
import github_release
import typer
from rich import print


class Workflow:
    """Class that defines an analysis workflow.

    Workflows can be defined by instantiating the workflow, then
    calling add_stage. For example,
        workflow = Workflow()
        workflow.add_stage(
            name="Stage 1",
            function=make_plot,
            output="cool_plot.pdf",
        )

    Note the workflow tries to be smart about not re-running rules
    and downloading cached outputs from Github releases.

    Note for the Github cache to work, you must have a Github personal access
    token saved under $GITHUB_TOKEN
    """

    def __init__(self) -> None:
        # Create an empty list of stages
        self.stages = []

        # Get the Github name
        self.git_repo = git.Repo(".")
        gh_url = self.git_repo.remotes.origin.url
        self.github_name = gh_url.removeprefix("git@github.com:").removesuffix(".git")

        # Start with an empty cache tag
        self.cache_tag = None
        self._tried_cache = False
        self._cache_error = False

        # Default to not verbose
        self.verbose = False

    def _connect_to_cache(self) -> None:
        """Connect to the remote cache."""
        # We will only run this once per session
        if self._tried_cache:
            return

        self._tried_cache = True
        try:
            # Get the list of all releases
            releases = github_release.get_releases(self.github_name)
            tags = [rel["tag_name"] for rel in releases]

            # If the tag is None and releases exist, default to most recent release
            if self.cache_tag is None and len(releases) > 0:
                self.cache_tag = tags[0]

            # If tag is still None, create default v1 tag
            if self.cache_tag is None:
                if self.verbose:
                    print("No releases exist in cache. Creating release v1.")
                github_release.gh_release_create(
                    self.github_name,
                    name="v1",
                    tag_name="v1",
                )
            # Otherwise if the tag is not in the list of releases, create release
            elif self.cache_tag not in tags:
                github_release.gh_release_create(
                    self.github_name,
                    name=self.cache_tag,
                    tag_name=self.cache_tag,
                )

            print(f"Using cache tag '{self.cache_tag}'")

        except Exception as exc:
            # Print the exception
            if self.verbose:
                print(exc)

            # We will proceed without the cache
            warnings.warn(
                "Failed to connect to remote cache. "
                "This could be due to any number of reasons including lack "
                "of internet connection, improper credentials, "
                "or non-existent Github repository. "
                "We will continue without the remote cache.",
                stacklevel=1,
            )
            self._cache_error = True

    def _get_cache_time(self, output: str | Path) -> float:
        """Get the timestamp at which this object was cached.

        Parameters
        ----------
        output : str or Path
            The output file for which to search the cache and return
            a cache time. If the file is not found, -99 is returned.

        Returns
        -------
        float
            The cache time in seconds. This is a Unix timestamp.
        """
        self._connect_to_cache()

        # If there is a known cache error, don't even try
        if self._cache_error:
            return -99

        # Otherwise check if this asset exists in the cache and return timestamp
        try:
            info = github_release.get_asset_info(
                self.github_name,
                self.cache_tag,
                output.name,
            )
            return datetime.fromisoformat(info["updated_at"]).timestamp()
        except Exception:
            return -99

    def _cache_output(self, output: str | Path | list) -> None:
        """Cache the output.

        Parameters
        ----------
        output : str or Path or list
            File or list of files to cache
        """
        self._connect_to_cache()

        # If there is a known cache error, don't try to cache anymore
        if self._cache_error:
            return

        # Recurse for lists of outputs
        if isinstance(output, (tuple, list)):
            for file in output:
                self._cache_output(file)
            return

        # Make sure the output is a Path object
        output = Path(output)

        # Get timestamp when asset was uploaded
        cache_mt = self._get_cache_time(output)

        # Get timestamp of local file creation
        local_mt = output.stat().st_mtime

        # If the local file isn't newer, we can skip
        if cache_mt >= local_mt:
            return

        # Otherwise we need to cache the output in an asset
        try:
            print(f"Caching '{output}'")

            # Get lists of assets
            asset_names = [
                asset["name"]
                for asset in github_release.get_assets(self.github_name, self.cache_tag)
            ]

            # If this asset is already in the cache, we must delete it first
            if output.name in asset_names:
                with contextlib.redirect_stdout(None):
                    github_release.gh_asset_delete(
                        self.github_name,
                        self.cache_tag,
                        output.name,
                    )

            # Now upload the asset
            with contextlib.redirect_stdout(None):
                github_release.gh_asset_upload(
                    self.github_name,
                    self.cache_tag,
                    str(output),
                )
        except Exception:
            pass

    def _download_cached_output(self, output: str | Path | list) -> None:
        """Download the cached output.

        Parameters
        ----------
        output : str or path or list
            File or list of files to download from the cache.
        """
        self._connect_to_cache()

        if self._cache_error:
            raise RuntimeError("Cannot connect to cache.")

        # Recurse for lists of outputs
        if isinstance(output, (tuple, list)):
            for file in output:
                self._download_cached_output(file)
            return

        # Make sure the output is a Path object
        output = Path(output)

        # Download the cached output
        with contextlib.chdir(output.parent), contextlib.redirect_stdout(None):
            github_release.gh_asset_download(
                self.github_name,
                self.cache_tag,
                str(output.name),
            )

        # Set the last modified time of the downloaded file to match the cache
        cache_time = self._get_cache_time(output)
        os.utime(output, times=(cache_time, cache_time))

    def _check_output_exists(self, stage_name: str, output: str | Path | list) -> None:
        """Check that the outputs of the stage exist.

        Parameters
        ----------
        stage_name : str
            The name of the stage
        output : str or path or list
            File or list of files to check.
        """
        # Recurse for lists of outputs
        if isinstance(output, (tuple, list)):
            for file in output:
                self._check_output_exists(file)
            return

        # Make sure the output is a Path object
        output = Path(output)

        if not output.exists():
            raise RuntimeError(
                f"Stage '{stage_name}' completed but output '{output}' is missing!"
            )

    def add_stage(
        self,
        name: str,
        function: Callable | None,
        output: str | Path | list,
        dependencies: str | list = None,
        cache: bool = False,
        **kwargs,
    ) -> None:
        """Add a stage to the workflow.

        Note stages are run in the order in which they are added to the
        workflow. This order might be important! For example, you can
        only list already-added stages in the `dependencies`.

        Parameters
        ----------
        name : str
            The name of the stage.
        function : Callable or None
            Function that defines the stage. This function must take the
            `output` keyword. Can also be None, in which case the output
            must be retrievable either from the local path or from the cache.
        output : str or Path or list
            The file(s) that are created by the stage.
        dependencies : str or list or None, default=None
            The stage(s) on which this stage depends.
        cache : bool, default=False
            Whether to cache the results of the stage.
        **kwargs
            Any other keywords to pass to the function.
        """
        # Check the name is unique:
        for stage in self.stages:
            if stage["name"] == name:
                raise ValueError(f"There is more than one stage with the name {name}.")

        # Check that any dependencies are already in the stage list
        if dependencies is None:
            pass
        elif isinstance(dependencies, (tuple, list)):
            for dep in dependencies:
                if not any(dep == stage["name"] for stage in self.stages):
                    raise ValueError(
                        f"Dependency '{dep}' for '{name}' not found. "
                        "Remember that the order in which you add stages does matter!"
                    )
        else:
            if not any(dependencies == stage["name"] for stage in self.stages):
                raise ValueError(
                    f"Dependency '{dependencies}' for '{name}' not found. "
                    "Remember that the order in which you add stages does matter!"
                )

        # Add the stage to the list
        self.stages.append(
            {
                "name": name,
                "function": function,
                "output": output,
                "dependencies": dependencies,
                "cache": cache,
                "kwargs": kwargs,
            }
        )

    def query_stages(self) -> dict:
        """Query current status of every stage.

        This is used to determine which stages need to be run.

        Returns
        -------
        dict
            Dictionary containing, for each stage:
            - "local": Whether the stage outputs exist locally
            - "cache": Whether the stage outputs exist in the cache
            - "newest": Which item is the newest (one of "local", "cache", "stage").
                If "stage" is the newest, that means the function that defines the
                stage has been updated more recently than the corresponding outputs,
                so the rule needs to be re-run.
        """
        # Create nested dictionary for every stage
        status = {stage["name"]: {} for stage in self.stages}

        # Loop over stages and save status indicators
        for stage in self.stages:
            name = stage["name"]

            # Check for file on local and in the cache
            if isinstance(stage["output"], (tuple, list)):
                status[name]["local"] = all(
                    [Path(file).exists() for file in stage["output"]]
                )

                status[name]["cache"] = all(
                    [self._get_cache_time(file) > -99 for file in stage["output"]]
                )
            else:
                status[name]["local"] = Path(stage["output"]).exists()
                status[name]["cache"] = self._get_cache_time(stage["output"]) > -99

            # Determine which item is newest
            if stage["function"] is None:
                stage_mt = -99
            else:
                stage_mt = Path(inspect.getfile(stage["function"])).stat().st_mtime
            if isinstance(stage["output"], (tuple, list)):
                # Get the local times
                local_mt = []
                for file in stage["output"]:
                    path = Path(file)
                    local_mt.append(-99 if not path.exists() else path.stat().st_mtime)
                local_mt = max(local_mt)

                # Get the cache times
                cache_mt = [self._get_cache_time(file) for file in stage["output"]]
                cache_mt = max(cache_mt)
            else:
                # Get the local time
                path = Path(stage["output"])
                local_mt = -99 if not path.exists() else path.stat().st_mtime

                # Get the cache time
                cache_mt = self._get_cache_time(stage["output"])

            # Handle cases where the local and cache do/don't exist
            if not status[name]["local"] and not status[name]["cache"]:
                status[name]["newest"] = "stage"
            elif status[name]["local"] and not status[name]["cache"]:
                if local_mt >= stage_mt:
                    status[name]["newest"] = "local"
                else:
                    status[name]["newest"] = "stage"
            elif not status[name]["local"] and status[name]["cache"]:
                if cache_mt >= stage_mt:
                    status[name]["newest"] = "cache"
                else:
                    status[name]["newest"] = "stage"
            else:
                if stage_mt > cache_mt and stage_mt > local_mt:
                    status[name]["newest"] = "stage"
                elif cache_mt > local_mt:
                    status[name]["newest"] = "cache"
                else:
                    status[name]["newest"] = "local"

        return status

    def run_stages(self) -> None:
        """Run the workflow."""
        # Query stage status
        status = self.query_stages()

        # Keep track of all re-run stages
        rerun_stages = []

        # Loop over stages
        for stage in self.stages:
            # Get the name of the stage
            name = stage["name"]

            # Check if dependencies have been re-run
            if stage["dependencies"] is None:
                dep_rerun = False
            elif isinstance(stage["dependencies"], (tuple, list)):
                dep_rerun = any([dep in rerun_stages for dep in stage["dependencies"]])
            else:
                dep_rerun = stage["dependencies"] in rerun_stages

            # If the cache is the newest version of output, download that version
            if status[name]["newest"] == "cache" and not dep_rerun:
                print(f"Downloading output for `{name}` from the cache")
                try:
                    self._download_cached_output(stage["output"])
                    continue
                except Exception:
                    print(
                        f"Failed to download output for '{name}' from the cache. "
                        "We will proceed without the cache."
                    )

            # If the local output is the newest version, skip this stage
            elif status[name]["newest"] == "local" and not dep_rerun:
                print(f"Skipping '{name}' because local output is up to date")
                if stage["cache"]:
                    self._cache_output(stage["output"])
                continue

            # If the function was None, and we couldn't find the output
            # either on the local or remote, then we have an error!
            elif stage["function"] is None:
                raise RuntimeError(
                    f"Stage '{name}' has a dummy function, but the output "
                    "was not found on the local path or in the cache."
                )

            # If we get this far, we need to run the stage
            # First we will print why the stage is being run
            elif status[name]["newest"] == "stage":
                print(f"Running '{name}' because the stage changed")
            elif dep_rerun:
                print(f"Running '{name}' because a dependency changed")
            else:
                raise RuntimeError("Edge case in the run_stages logic!")

            # Run the stage
            stage["function"](output=stage["output"], **stage["kwargs"])

            # Check that the outputs now exist
            self._check_output_exists(name, stage["output"])

            # Cache the results
            if stage["cache"]:
                self._cache_output(stage["output"])

            # Save this stage in the list of run stages
            rerun_stages.append(name)

        print("Workflow completed!")

    def cli(self) -> None:
        """Create command-line interface for the workflow."""
        # Create a Typer app
        app = typer.Typer(add_completion=False, no_args_is_help=True)

        # Add global options
        @app.callback()
        def main(verbose: bool = False):
            """Command line interface for a workflow"""
            self.verbose = verbose

        # Define function to print the stage query
        @app.command()
        def query_stages():
            status = self.query_stages()
            for stage in status:
                print(f"'{stage}' : {status[stage]}")

        query_stages.__doc__ = self.query_stages.__doc__.replace("\n\n", "\n\n\b")

        # Define function to run stages
        @app.command()
        def run_stages():
            self.run_stages()

        run_stages.__doc__ = self.run_stages.__doc__.replace("\n\n", "\n\n\b")

        # Run CLI
        app()
