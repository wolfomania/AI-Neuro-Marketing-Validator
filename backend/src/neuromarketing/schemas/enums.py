from enum import StrEnum


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(StrEnum):
    VALIDATING = "validating"
    EXTRACTING_EVENTS = "extracting_events"
    TRIBE_PREDICT = "tribe_predict"
    SUMMARIZING_ROIS = "summarizing_rois"
    GENERATING_PLOTS = "generating_plots"
    ANALYZING_WITH_CLAUDE = "analyzing_with_claude"
    SAVING_RESULTS = "saving_results"


STAGE_LABELS: dict[ProcessingStage, str] = {
    ProcessingStage.VALIDATING: "Validating video...",
    ProcessingStage.EXTRACTING_EVENTS: "Extracting audio/visual events...",
    ProcessingStage.TRIBE_PREDICT: "Predicting brain activations...",
    ProcessingStage.SUMMARIZING_ROIS: "Mapping brain regions...",
    ProcessingStage.GENERATING_PLOTS: "Generating brain visualizations...",
    ProcessingStage.ANALYZING_WITH_CLAUDE: "AI analyzing neural patterns...",
    ProcessingStage.SAVING_RESULTS: "Saving results...",
}

STAGE_PROGRESS: dict[ProcessingStage, int] = {
    ProcessingStage.VALIDATING: 5,
    ProcessingStage.EXTRACTING_EVENTS: 15,
    ProcessingStage.TRIBE_PREDICT: 50,
    ProcessingStage.SUMMARIZING_ROIS: 60,
    ProcessingStage.GENERATING_PLOTS: 70,
    ProcessingStage.ANALYZING_WITH_CLAUDE: 85,
    ProcessingStage.SAVING_RESULTS: 95,
}
