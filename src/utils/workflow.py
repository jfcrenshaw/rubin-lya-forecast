"""Define the workflow class"""

import contextlib
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import ContextManager

import git
import github_release
import humanize
import typer
from rich import print
import numpy as np
from abc import ABC, abstractmethod
from types import SimpleNamespace
from inspect import getfile


class Stage(ABC):
    """Abstract base class for workflow stages.

    You must subclass and define the '_run' method.
    """

    def __init__(
        self,
        name: str,
        output: str | Path | list,
        dependencies: str | list = None,
        cache: bool = False,
        paths: SimpleNamespace | None = None,
        wf_vars: SimpleNamespace | None = None,
        **kwargs,
    ) -> None:
        """Create stage.

        Parameters
        ----------
        name : str
            The name of the stage
        output : str | Path | list
            The file(s) that are created by the stage
        dependencies : str | list, default=None
            The stage(s) on which this stage depends
        cache : bool, default=False
            Whether to cache the results of the stage
        paths : SimpleNamespace or None, default=None
            Namespace object that defines paths for saving,
            loading objects.
        wf_vars : SimpleNamespace or None, default=None
            Namespace object for global workflow variables
        **kwargs
            Any other keywords to pass to the function.
        """
        # Save the parameters
        self.name = name
        self.output = output
        self.dependencies = dependencies
        self.cache = cache
        self.wf_vars = SimpleNamespace() if wf_vars is None else wf_vars
        self.stage_vars = SimpleNamespace(**kwargs)

        # If no paths provided, just use root path
        if paths is None:
            git_repo = git.Repo(".", search_parent_directories=True)
            self.paths = SimpleNamespace(root=Path(git_repo.working_tree_dir))
        else:
            self.paths = paths

        # Start with empty resolution status
        self.resolution = None

    @property
    def _output_list(self) -> list:
        """Return the output in a list."""
        # Get the output
        output = self.output

        # Convert to list
        if isinstance(output, list):
            pass
        elif isinstance(output, tuple):
            output = list(output)
        else:
            output = [output]

        # Make sure every element is a Path
        output = [Path(file) for file in output]

        return output

    def _prep_output_destination(self) -> None:
        """Create the output parent directory if it does not exist."""
        # Get output list
        output = self._output_list

        # Loop over files
        for file in output:
            if file.parent.exists():
                continue
            if self.verbose:
                print(f"Creating parent directory for output '{file}'")
            file.parent.mkdir(parents=True)

    def _check_output_exists(self, output: str | Path | None = None) -> bool:
        """Return whether the output exists locally.

        Parameters
        ----------
        output: str or Path or None, default=None
            The output to check existence for. If None, all outputs
            are checked.

        Returns
        -------
        bool
            Whether the output exits locally
        """
        # Determine the output to query
        if output is None:
            output = self._output_list
        else:
            output = [output]

        return all([file.exists() for file in output])

    def _get_local_time(self, output: str | Path | None = None) -> float:
        """Get the local timestamp for the output.

        Parameters
        ----------
        output: str or Path or None, default=None
            The output to get the local time for. If None, the
            minimum local time for all outputs is return.

        Returns
        -------
        float
            The local time in seconds. This is a Unix timestamp.
        """
        if not self._check_output_exists():
            return -99

        # Determine the output to query
        if output is None:
            output = self._output_list
        else:
            output = [output]

        return min([file.stat().st_mtime for file in output])

    def _get_cache_time(
        self,
        workflow: "Workflow | None",
        output: str | Path | None = None,
    ) -> float:
        """Get the cache timestamp for the output.

        Parameters
        ----------
        workflow: Workflow or None
            Workflow with a cache
        output: str or Path or None, default=None
            The output to get the cache time for. If None, the
            minimum cache time for all outputs is return.

        Returns
        -------
        float
            The cache time in seconds. This is a Unix timestamp.
        """
        # If no workflow, don't even try
        if workflow is None:
            return -99

        # Connect to workflow cache
        workflow._connect_to_cache()

        # If cache not connected, don't even try
        if not workflow._cache_connected:
            return -99

        # Determine the output to query
        if output is None:
            output = self._output_list
        else:
            output = [output]

        # Return time stamps for files
        try:
            times = []
            for file in output:
                info = github_release.get_asset_info(
                    workflow.github_name,
                    workflow.cache_tag,
                    file.name,
                )
                times.append(datetime.fromisoformat(info["updated_at"]).timestamp())
            return min(times)
        except Exception:
            return -99

    def _check_output_cache_exists(
        self,
        workflow: "Workflow | None",
        output: str | Path | None = None,
    ) -> bool:
        """Check whether the output exists in the cache.

        Parameters
        ----------
        workflow: Workflow or None
            Workflow with a cache
        output: str or Path or None, default=None
            The output to check existence for. If None, all outputs
            are checked.

        Returns
        -------
        bool
            Whether the output exists in the cache
        """
        return self._get_cache_time(workflow, output) > -99

    def _cache_output(self, workflow: "Workflow | None") -> None:
        """Cache the output.

        Parameters
        ----------
        workflow: Workflow or None
            Workflow with a cache
        """
        # If no workflow, don't even try
        if workflow is None:
            return

        # Connect to workflow cache
        workflow._connect_to_cache()

        # If cache not connected, don't even try
        if not workflow._cache_connected:
            return

        # Get output list
        output = self._output_list

        # Get lists of cache assets
        asset_names = [
            asset["name"]
            for asset in github_release.get_assets(
                workflow.github_name, workflow.cache_tag
            )
        ]

        # Loop over every file
        for file in output:
            # Get the timestamps
            local_mt = self._get_local_time(file)
            cache_mt = self._get_cache_time(workflow, file)

            # Skip if cache up-to-date
            if cache_mt >= local_mt:
                continue

            # Otherwise we need to cache the output in an asset
            print(f"Caching '{file}'")

            # If asset already in cache, we must delete it first
            if file.name in asset_names:
                with workflow._get_print_context():
                    github_release.gh_asset_delete(
                        workflow.github_name,
                        workflow.cache_tag,
                        file.name,
                    )

            # Now upload asset
            with workflow._get_print_context():
                github_release.gh_asset_upload(
                    workflow.github_name,
                    workflow.cache_tag,
                    str(file),
                )

            # Set local timestamp to match cache
            cache_time = self._get_cache_time(workflow, file)
            os.utime(file, times=(cache_time, cache_time))

    def _download_cached_output(self, workflow: "Workflow | None") -> None:
        """Download the cached output.

        Parameters
        ----------
        workflow : Workflow or None
            Workflow with a cache
        """
        # If no workflow,raise error
        if workflow is None:
            raise RuntimeError("No workflow provided.")

        # Connect to workflow cache
        workflow._connect_to_cache()

        # If cache not connected, don't even try
        if not workflow._cache_connected:
            raise RuntimeError("Cannot connect to cache.")

        # Get output list
        output = self._output_list

        # Make sure the output parent directory exists
        self._prep_output_destination()

        # Loop over files
        for file in output:
            # Download the cached file
            with contextlib.chdir(file.parent), workflow._get_print_context():
                github_release.gh_asset_download(
                    workflow.github_name,
                    workflow.cache_tag,
                    str(file.name),
                )

            # Set the last modified time of the downloaded file to match the cache
            cache_time = self._get_cache_time(workflow, file)
            os.utime(file, times=(cache_time, cache_time))

    def _get_stage_time(self) -> float:
        """Get the timestamp of the stage definition.

        Returns
        -------
        float
            Last time the file defining the stage changed.
            This is a Unix timestamp.
        """
        file = Path(getfile(self.__class__))
        return file.stat().st_mtime

    def query(self, workflow: "Workflow | None" = None) -> dict:
        """This is used to determine if the stage needs to be run.

        Parameters
        ----------
        workflow: Workflow or None, default=None
            A Workflow with a cache

        Returns
        -------
            dict
                Dictionary containing the following information:
                - "local": Whether the stage outputs exist locally
                - "cache": Whether the stage outputs exist in the cache
                - "newest": Which item is newest (one of "local", "cache", "stage").
                    If "stage" is newest, that means the file that defines the stage
                    has been updated more recently than the corresponding outputs,
                    so the rule needs to be re-run.
        """
        # Check existence
        exist_local = self._check_output_exists()
        exist_cache = self._check_output_cache_exists(workflow)

        # Get all the time stamps
        times = [
            self._get_local_time(),
            self._get_cache_time(workflow),
            self._get_stage_time(),
        ]
        newest = ["local", "cache", "stage"][np.argmax(times)]

        return {
            "local": exist_local,
            "cache": exist_cache,
            "newest": newest,
        }

    @staticmethod
    def split_seed(seed: int, N: int = 2, max_seed: int = int(1e12)) -> np.ndarray:
        """Split the random seed.

        Parameters
        ----------
        seed : int
            The initial seed
        N : int, default=2
            The number of seeds to produce
        max_seed : int, default=1e12
            The maximum allowed value for seeds
        """
        rng = np.random.default_rng(seed)
        return rng.integers(0, max_seed, size=N)

    def _pre_run(self, workflow: "Workflow | None", dep_changed: bool) -> None:
        """Execute pre-run steps.

        Parameters
        ----------
        workflow: Workflow or None
            A workflow with wf_vars and a cache
        dep_changed: bool, default=False
            Whether one of the stage dependencies changed. If True,
            the stage will be re-run regardless of the status.
        """
        # Query the stage status to determine if we need to re-run
        status = self.query(workflow)

        # If local output exists and it is newest, don't do anything
        if status["newest"] == "local" and not dep_changed:
            print(f"Skipping '{self.name}' because local output is up to date")
            self.resolution = "local"
            if self.cache:
                self._cache_output(workflow)

        # If cached output exists and it is newest, download it
        elif status["newest"] == "cache" and not dep_changed:
            print(f"Downloading output for `{self.name}` from the cache")
            try:
                self._download_cached_output(workflow)
                self.resolution = "cache"
            except Exception:
                print(
                    f"Failed to download output for '{self.name}' from the cache. "
                    "We will proceed without the cache and re-run the stage."
                )

        # If this is a DummyStage and we couldn't find the output, raise error
        elif isinstance(self, DummyStage):
            raise RuntimeError(
                f"'{self.name}' is a DummyStage, but the output "
                "was not found on the local path or in the cache."
            )

        # If the stage is not resolved, we will need to run it
        if self.resolution is None:
            # First print why we are running the stage
            if not status["local"] and not status["cache"]:
                print(f"Running '{self.name}' because the output does not exist")
            elif status["newest"] == "cache":
                print(f"Running '{self.name}' because cache download failed")
            elif status["newest"] == "stage":
                print(f"Running '{self.name}' because the stage changed")
            elif dep_changed:
                print(f"Running '{self.name}' because a dependency changed")
            else:
                raise RuntimeError("Edge case in run logic!")

    @abstractmethod
    def _run(self) -> None:
        """Run the stage.

        Subclass must define this stage. It must take no parameters
        and return None.
        """
        ...

    def _post_run(self) -> None:
        """Execute post-run steps."""
        for file in self._output_list:
            if not self._check_output_exists(file):
                raise RuntimeError(
                    f"Stage '{self.name}' completed but output '{file}' is missing!"
                )

    def run(
        self,
        workflow: "Workflow | None" = None,
        dep_changed: bool = False,
    ) -> None:
        """
        Run the stage.

        Parameters
        ----------
        workflow: Workflow or None
            A workflow with wf_vars and a cache. If None, the pre- and
            post-run methods are not run, and the main '_run' method
            is guaranteed to run.
        dep_changed: bool, default=False
            Whether one of the stage dependencies changed. If True,
            the stage will be re-run regardless of the status.
        """
        # Execute pre-run steps
        if workflow is not None:
            self._pre_run(workflow, dep_changed)

        # Run the stage
        if self.resolution is None or workflow is None:
            # Make sure the output parent directories exist
            self._prep_output_destination()

            # Run the main algorithm
            self._run()

            # Indicate that this stage was run
            self.resolution = "run"

        # Execute post-run steps
        if workflow is not None:
            self._post_run()


class DummyStage(Stage):
    """Dummy stage that does nothing.

    This can be used for syncing input files to/from the cache.
    """

    def _get_stage_time(self) -> float:
        """Get the timestamp of the stage definition.

        We need to redefine this so it is always older than the
        local and cache values.

        Returns
        -------
        float
            -100
        """
        return -100

    def _run(self) -> None:
        """Don't do anything!"""
        pass


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

    def __init__(self, **kwargs) -> None:
        """Create workflow.

        Parameters
        ----------
        **kwargs
            Any keyword arguments to add to the workflow global variables.
        """
        # Create an empty list of stages
        self.stages = []

        # Get git and github info
        self.git_repo = git.Repo(".", search_parent_directories=True)
        gh_url = self.git_repo.remotes.origin.url
        self.github_name = gh_url.removeprefix("git@github.com:").removesuffix(".git")

        # Start with an empty cache tag
        self.cache_tag = None
        self._cache_tags = []
        self._cache_connected = None

        # Create the root paths object
        self.paths = SimpleNamespace(root=Path(self.git_repo.working_tree_dir))

        # Save the workflow variables
        self.wf_vars = SimpleNamespace(**kwargs)

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
        stage: Stage | None,
        output: str | Path | list,
        dependencies: str | list = None,
        cache: bool = False,
        paths: SimpleNamespace | None = None,
        wf_vars: SimpleNamespace | None = None,
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
        stage : Stage or None
            Class that defines the stage. If None, DummyStage is used.
        output : str or Path or list
            The file(s) that are created by the stage.
        dependencies : str or list or None, default=None
            The stage(s) on which this stage depends.
        cache : bool, default=False
            Whether to cache the results of the stage.
        paths : SimpleNamespace or None, default=None
            Override the workflow paths. If None, the workflow
            passes the current paths object.
        wf_vars : SimpleNamespace or None, default=None
            Override the global workflow variables. If None,
            the workflow passes the current global variables.
        **kwargs
            Any other keywords to pass to the function.
        """
        # Check the name is unique:
        for prev_stage in self.stages:
            if name == prev_stage.name:
                raise ValueError(
                    f"There is more than one stage with the name '{name}'."
                )

        # Check that any dependencies are already in the stage list
        if dependencies is None:
            pass
        elif isinstance(dependencies, (tuple, list)):
            for dep in dependencies:
                if not any(dep == stage.name for stage in self.stages):
                    raise ValueError(
                        f"Dependency '{dep}' for '{name}' not found. "
                        "Remember the order in which you add stages does matter!"
                    )
        else:
            if not any(dependencies == stage.name for stage in self.stages):
                raise ValueError(
                    f"Dependency '{dependencies}' for '{name}' not found. "
                    "Remember the order in which you add stages does matter!"
                )

        # If stage is None, use DummyStage
        if stage is None:
            stage = DummyStage
        elif not issubclass(stage, Stage):
            raise TypeError(f"Stage for '{name}' is not a stage!")

        # Determine the paths and workflow variables to pass
        if paths is None:
            paths = self.paths
        if wf_vars is None:
            wf_vars = SimpleNamespace(**self.wf_vars.__dict__.copy())

        # Add the stage to the list
        self.stages.append(
            stage(
                name=name,
                output=output,
                dependencies=dependencies,
                cache=cache,
                paths=paths,
                wf_vars=wf_vars,
                **kwargs,
            )
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
        # Get the status of the workflow
        status = {stage.name: stage.query() for stage in self.stages}

        # Print status
        if self.verbose or _cli_print:
            print("Stage status:")
            for stage in status:
                print(f"'{stage}': {status[stage]}")
            print()

        return status

    def run(self) -> None:
        """Run the workflow."""
        # Keep track of all re-run stages
        rerun_stages = []

        for stage in self.stages:
            # Check if dependencies have been re-run
            if stage.dependencies is None:
                dep_changed = False
            elif isinstance(stage.dependencies, (tuple, list)):
                dep_changed = any([dep in rerun_stages for dep in stage.dependencies])
            else:
                dep_changed = stage.dependencies in rerun_stages

            # Run the stage
            stage.run(
                workflow=self,
                dep_changed=dep_changed,
            )

            # If this stage was run, save it in the re-run list
            if stage.resolution == "run":
                rerun_stages.append(stage.name)

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
