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
WANDB_OUTPUT_DIR_PATH="${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}/wandb/"
WANDB_PROJECT_DIR_PATH="${WANDB_OUTPUT_DIR_PATH}"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Loop through all matching offline run directories.
# Note: No quotes around the wildcard pattern, because we want to expand it.
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Note: We usually do not need to sync the online runs,
# because they are already synced automatically.

# Exit with the exit status of the last command
exit $?
