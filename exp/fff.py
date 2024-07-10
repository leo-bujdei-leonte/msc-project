"""
Source: https://cloud.google.com/vertex-ai/docs/
"""

from typing import Any

from google.cloud import aiplatform

def create_experiment_run_sample(
    experiment_name: str,
    run_name: str,
    experiment_run_tensorboard: str | aiplatform.Tensorboard | None,
    project: str,
    location: str,
):
    aiplatform.init(experiment=experiment_name, project=project, location=location)
    aiplatform.start_run(run=run_name, tensorboard=experiment_run_tensorboard)

def end_experiment_run_sample(
    experiment_name: str,
    run_name: str,
    project: str,
    location: str,
):
    aiplatform.init(experiment=experiment_name, project=project, location=location)
    aiplatform.start_run(run=run_name, resume=True)
    aiplatform.end_run()

def resume_experiment_run_sample(
    experiment_name: str,
    run_name: str,
    project: str,
    location: str,
):
    aiplatform.init(experiment=experiment_name, project=project, location=location)

    aiplatform.start_run(run=run_name, resume=True)

def delete_experiment_run_sample(
    run_name: str,
    experiment: str | aiplatform.Experiment,
    project: str,
    location: str,
    delete_backing_tensorboard_run: bool = False,
):
    experiment_run = aiplatform.ExperimentRun(
        run_name=run_name, experiment=experiment, project=project, location=location
    )

    experiment_run.delete(delete_backing_tensorboard_run=delete_backing_tensorboard_run)

def update_experiment_run_state_sample(
    run_name: str,
    experiment: str | aiplatform.Experiment,
    project: str,
    location: str,
    state: aiplatform.gapic.Execution.State,
) -> None:
    experiment_run = aiplatform.ExperimentRun(
        run_name=run_name,
        experiment=experiment,
        project=project,
        location=location,
    )

    experiment_run.update_state(state)

def log_pipeline_job_to_experiment_sample(
    experiment_name: str,
    pipeline_job_display_name: str,
    template_path: str,
    pipeline_root: str,
    project: str,
    location: str,
    parameter_values: dict[str | Any] | None = None,
):
    aiplatform.init(project=project, location=location)

    pipeline_job = aiplatform.PipelineJob(
        display_name=pipeline_job_display_name,
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
    )

    pipeline_job.submit(experiment=experiment_name)

