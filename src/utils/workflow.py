"""Define the workflow class"""

import contextlib
import inspect
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, ContextManager

import git
import github_release
import humanize
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

        # Get git and github info
        self.git_repo = git.Repo(".")
        gh_url = self.git_repo.remotes.origin.url
        self.github_name = gh_url.removeprefix("git@github.com:").removesuffix(".git")

        # Start with an empty cache tag
        self.cache_tag = None
        self._cache_tags = []
        self._cache_connected = None

        # Default to not verbose
        self.verbose = False

    def _get_print_context(self) -> ContextManager:
        """Get the print context for the verbosity level.

        Returns
        -------
        ContextManager
            A context manager that hides print statements if self.verbose==False
        """
        if self.verbose:
            return contextlib.nullcontext()
        else:
            return contextlib.redirect_stdout(None)

    @staticmethod
    def _no_connection_warning() -> None:
        """Warn about the lack of cache connection."""
        warnings.warn(
            "Failed to connect to remote cache. "
            "This could be due to any number of reasons including lack "
            "of internet connection, improper credentials, "
            "or non-existent Github repository.",
            stacklevel=2,
        )

    def _connect_to_cache(self) -> None:
        """Connect to the remote cache."""
        # We will only run this once per session
        if self._cache_connected is not None:
            return

        # Check we can connect to Github
        try:
            github_release.get_refs(self.github_name)
            self._cache_connected = True
        except Exception as exc:
            self._cache_connected = False

            # If no cache tag was provided, just return
            if self.cache_tag is None:
                return

            # Otherwise we want to want about the lack of connection
            if self.verbose:
                print(exc)
            self._no_connection_warning()
            return

        # Get list of existing cache tags
        releases = github_release.get_releases(self.github_name)
        tags = [rel["tag_name"] for rel in releases]

        # If cache_tag not in list of existing tags, create new release
        if self.cache_tag is None:
            pass
        elif self.cache_tag not in tags:
            print(f"Creating new cache tag '{self.cache_tag}'")
            with self._get_print_context():
                github_release.gh_release_create(
                    self.github_name,
                    name=self.cache_tag,
                    tag_name=self.cache_tag,
                )
            tags = [self.cache_tag] + tags
        # Otherwise use existing release
        else:
            print(f"Using existing cache tag '{self.cache_tag}'")

        # Save the list of cache tags
        self._cache_tags = tags

        # Print the cache info
        if self.verbose:
            self.query_cache(include_assets=False)

        # Add a blank line after cache info
        print()

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

        # If we are not connected to the cache, don't even try
        if not self._cache_connected:
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

        # If we are not connected to the cache, don't even try
        if not self._cache_connected:
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
                with self._get_print_context():
                    github_release.gh_asset_delete(
                        self.github_name,
                        self.cache_tag,
                        output.name,
                    )

            # Now upload the asset
            with self._get_print_context():
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

        # If we are not connected to the cache, don't even try
        if not self._cache_connected:
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

    def get_existing_cache_tags(self, _cli_print: bool = False) -> list:
        """Get list of existing cache tags.

        Returns
        -------
        list
            List of existing cache tags
        """
        self._connect_to_cache()
        tags = self._cache_tags.copy()

        if len(tags) == 0 and not self._cache_connected:
            self._no_connection_warning()

        if self.verbose or _cli_print:
            print("Existing cache tags:")
            print(tags)

        return tags

    def query_cache(
        self,
        include_assets: bool = True,
        _cli_print: bool = False,
    ) -> dict:
        """Query the cache.

        Parameters
        ----------
        include_assets : bool
            Whether to include asset info.

        Returns
        -------
        dict
            Dictionary containing cache info
        """
        self._connect_to_cache()

        # Get the cache info
        if not self._cache_connected:
            info = {}
        else:
            info = github_release.get_release_info(self.github_name, self.cache_tag)

        # Remove asset info it it's not wanted
        if not include_assets:
            info.pop("assets")

        # Print a nicely formatted summary of key information
        if self.verbose or _cli_print:
            try:
                author = info["author"]["login"]
            except Exception:
                author = None
            print(f"Release '{info.get('name')}' info")
            print(f"  {'Tag name':<13}: {info.get('tag_name')}")
            print(f"  {'ID':<13}: {info.get('id')}")
            print(f"  {'Created':<13}: {info.get('created_at')}")
            print(f"  {'Author':<13}: {author}")
            print(f"  {'Is published':<13}: {not info.get('draft')}")
            print(f"  {'URL':<13}: {info.get('html_url')}")

            for i, asset in enumerate(info.get("assets", [])):
                print()
                print(f"  Asset #{i}")
                print(f"    {'name':<8}: {asset['name']}")
                print(f"    {'created':<8}: {asset['created_at']}")
                print(f"    {'updated':<8}: {asset['updated_at']}")
                print(f"    {'author':<8}: {asset['uploader']['login']}")
                print(f"    {'size':<8}: {humanize.naturalsize(asset['size'])}")
                print(f"    {'url':<8}: {asset['browser_download_url']}")

        return info

    def delete_cache(self, confirm: bool) -> None:
        """Delete the cache saved under this tag.

        Parameters
        ----------
        confirm : bool
            A boolean that must be set to true in order to delete the cache.
            This is to provide an extra safety check before deleting.
        """
        if not confirm:
            print("Not deleting cache because confirm==False.")
            return

        self._connect_to_cache()
        try:
            print(f"Deleting cache with tag '{self.cache_tag}'")
            with self._get_print_context():
                github_release.gh_release_delete(self.github_name, self.cache_tag)
            self._cache_tags.remove(self.cache_tag)
        except Exception as exc:
            if self.verbose:
                print(exc)
            print(f"Could not delete cache with tag '{self.cache_tag}'")

    def delete_all_caches(self, confirm: bool) -> None:
        """Delete all caches for this workflow.

        Parameters
        ----------
        confirm : bool
            A boolean that must be set to true in order to delete the cache.
            This is to provide an extra safety check before deleting.
        """
        if not confirm:
            print("Not deleting caches because confirm==False.")
            return

        self._connect_to_cache()
        try:
            print("Deleting all caches")
            for tag in self.get_existing_cache_tags():
                with self._get_print_context():
                    github_release.gh_release_delete(self.github_name, tag)
                self._cache_tags.remove(tag)
        except Exception as exc:
            if self.verbose:
                print(exc)
            print("Could not delete all caches")

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

    def query_stages(self, *, _cli_print: bool = False) -> dict:
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

        # Print status
        if self.verbose or _cli_print:
            print("Stage status:")
            for stage in status:
                print(f"'{stage}': {status[stage]}")
            print()

        return status

    def run(self) -> None:
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
            if not status[name]["local"] and not status[name]["cache"]:
                print(f"Running '{name}' because the output does not exist.")
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

        print("\nWorkflow completed!")

    def cli(self) -> None:
        """Create command-line interface for the workflow."""
        # Create a Typer app
        app = typer.Typer(add_completion=False, no_args_is_help=True)

        # Add global options
        @app.callback()
        def main(verbose: bool = False):
            """Command line interface for a workflow"""
            self.verbose = verbose

        # Command to list existing cache tags
        @app.command()
        def get_existing_cache_tags(include_assets: bool = True):
            self.get_existing_cache_tags(_cli_print=True)

        get_existing_cache_tags.__doc__ = self.get_existing_cache_tags.__doc__.replace(
            "\n\n", "\n\n\b"
        )

        # Command to delete cache
        @app.command()
        def delete_cache(confirm: bool) -> None:
            self.delete_cache(confirm=confirm)

        delete_cache.__doc__ = self.delete_cache.__doc__.replace("\n\n", "\n\n\b")

        # Command to delete all caches
        @app.command()
        def delete_all_caches(confirm: bool) -> None:
            self.delete_all_caches(confirm=confirm)

        delete_all_caches.__doc__ = self.delete_all_caches.__doc__.replace(
            "\n\n", "\n\n\b"
        )

        # Command to print cache query
        @app.command()
        def query_cache(include_assets: bool = True) -> None:
            self.query_cache(include_assets=include_assets, _cli_print=True)

        query_cache.__doc__ = self.query_cache.__doc__.replace("\n\n", "\n\n\b")

        # Command to print stage query
        @app.command()
        def query_stages() -> None:
            self.query_stages(_cli_print=True)

        query_stages.__doc__ = self.query_stages.__doc__.replace("\n\n", "\n\n\b")

        # Command to run stages
        @app.command()
        def run() -> None:
            self.run()

        run.__doc__ = self.run.__doc__.replace("\n\n", "\n\n\b")

        # Run CLI
        app()
