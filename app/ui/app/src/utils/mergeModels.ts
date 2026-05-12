import { Model } from "@/gotypes";

// Featured local models list (in priority order). Eliza Cloud models (`*:cloud`) append when enabled.
export const FEATURED_MODELS = [
  "gpt-oss:120b",
  "gpt-oss:20b",
  "gemma4:31b",
  "gemma4:26b",
  "gemma4:e4b",
  "gemma4:e2b",
  "deepseek-r1:8b",
  "qwen3-coder:30b",
  "qwen3-vl:30b",
  "qwen3-vl:8b",
  "qwen3-vl:4b",
  "qwen3.5:27b",
  "qwen3.5:9b",
  "qwen3.5:4b",
];

function alphabeticalSort(a: Model, b: Model): number {
  return a.model.toLowerCase().localeCompare(b.model.toLowerCase());
}

// Merges models for the picker: featured locals first, then remaining locals, then Eliza cloud models.
export function mergeModels(
  localModels: Model[],
  hideCloudModels: boolean = false,
): Model[] {
  const allModels = (localModels || []).map((model) => model);

  const cloudModelsList = hideCloudModels
    ? []
    : allModels.filter((m) => m.isCloud());

  const localsOnly = allModels.filter((m) => !m.isCloud());

  const featuredModels = FEATURED_MODELS.map((model) => {
    const localMatch = localsOnly.find(
      (m) => m.model.toLowerCase() === model.toLowerCase(),
    );

    if (localMatch) return localMatch;

    return new Model({
      model,
    });
  });

  const remainingModels = localsOnly.filter(
    (model) =>
      !FEATURED_MODELS.some(
        (f) => f.toLowerCase() === model.model.toLowerCase(),
      ),
  );

  featuredModels.sort(
    (a, b) =>
      FEATURED_MODELS.indexOf(a.model) - FEATURED_MODELS.indexOf(b.model),
  );

  remainingModels.sort(alphabeticalSort);

  cloudModelsList.sort(alphabeticalSort);

  return [...featuredModels, ...remainingModels, ...cloudModelsList];
}
