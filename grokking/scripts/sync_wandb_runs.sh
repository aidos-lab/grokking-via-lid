#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # #
# Default values
DRY_RUN=false

# Name of the environment variables holding the repository base path
THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME="GROKKING_REPOSITORY_BASE_PATH"

# # # # # # # # # # # # # # # # # # # # # # # # #
# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --dry-run)
    DRY_RUN=true
    shift # Remove --dry-run from processing
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

# # # # # # # # # # # # # # # # # # # # # # # #
# Check if required variables are set

# Check if the environment variable is set
if [[ -z "${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}" ]]; then
  echo "@@@ Error: Environment variable ${THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME} is not set."
  exit 1
fi

# Construct the full path to the project directory
WANDB_OUTPUT_DIR_PATH="${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}/wandb"

WANDB_PROJECT_NAMES=(
  "grokking_replica"
  "grokking_replica_HHU_Hilbert_HPC_runs_different_dataset_portions_long_large_p_new_topological_analysis"
  "grokking_replica_HHU_Hilbert_HPC_runs_different_dataset_portions_long_new_topological_analysis"
  "grokking_replica_HHU_Hilbert_HPC_runs_different_dataset_portions_long"
  "grokking_replica_HHU_Hilbert_HPC_runs_different_operations_and_p_values_long_with_regularization"
  "grokking_replica_HHU_Hilbert_HPC_runs_different_operations_and_p_values_long"
  "grokking_replica_HHU_Hilbert_HPC_runs_very_long"
)

for WANDB_PROJECT_NAME in "${WANDB_PROJECT_NAMES[@]}"; do
  # Note:
  # - There is another 'wandb' directory inside the project directory, this is not a mistake.
  WANDB_PROJECT_DIR_PATH="${WANDB_OUTPUT_DIR_PATH}/${WANDB_PROJECT_NAME}/wandb/"

  echo ">>> Syncing Weights and Biases runs for project ${WANDB_PROJECT_NAME} ..."
  echo ">>> Checking for runs in ${WANDB_PROJECT_DIR_PATH} ..."

  # Check if the directory exists
  if [ ! -d "$WANDB_PROJECT_DIR_PATH" ]; then
    echo "@@@ Error: Directory ${WANDB_PROJECT_DIR_PATH} does not exist."
    echo "@@@ Skipping syncing for the project ${WANDB_PROJECT_NAME}."

    # Skip to the next project
    continue
  fi

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # Loop through all matching offline run directories.
  #
  # Notes:
  # - No quotes around the wildcard pattern, because we want to expand it.
  # - We usually do not need to sync the online runs,
  #   because they are already uploaded automatically on machines with internet access.
  for OFFLINE_RUN_DIR in "${WANDB_PROJECT_DIR_PATH}"/offline-*; do
    # Check if any matches were found
    if [ -d "$OFFLINE_RUN_DIR" ]; then
      if [ "$DRY_RUN" = true ]; then
        echo ">>> [DRY RUN] Would sync: ${OFFLINE_RUN_DIR}"
      else
        echo ">>> Syncing ${OFFLINE_RUN_DIR} to Weights and Biases ..."

        wandb sync \
          --include-offline \
          "${OFFLINE_RUN_DIR}"

        # Exit if the sync command fails for any reason
        if [ $? -ne 0 ]; then
          echo "@@@ Error: Sync failed for ${OFFLINE_RUN_DIR}"
          exit 1
        fi

        echo ">>> Syncing ${OFFLINE_RUN_DIR} to Weights and Biases DONE"
      fi
    fi
  done
done

# Exit with the exit status of the last command
exit $?
