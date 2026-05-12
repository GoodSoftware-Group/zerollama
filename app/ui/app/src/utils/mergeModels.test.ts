import { describe, it, expect } from "vitest";
import { Model } from "@/gotypes";
import { mergeModels, FEATURED_MODELS } from "@/utils/mergeModels";
import "@/api";

describe("Model merging logic", () => {
  it("should put featured locals first, then remaining locals, then cloud models", () => {
    const localModels: Model[] = [
      new Model({ model: "gpt-oss:120b-cloud" }),
      new Model({ model: "llama3:latest" }),
      new Model({ model: "mistral:latest" }),
    ];

    const merged = mergeModels(localModels, false);

    expect(merged[merged.length - 1].model).toBe("gpt-oss:120b-cloud");

    for (let i = 0; i < FEATURED_MODELS.length; i++) {
      expect(merged[i].model).toBe(FEATURED_MODELS[i]);
    }

    const featuredCount = FEATURED_MODELS.length;
    expect(merged[featuredCount].model).toBe("llama3:latest");
    expect(merged[featuredCount + 1].model).toBe("mistral:latest");

    expect(merged.length).toBe(FEATURED_MODELS.length + 3);
  });

  it("should omit cloud models when hideCloudModels is true", () => {
    const localModels: Model[] = [
      new Model({ model: "gpt-oss:120b-cloud" }),
      new Model({ model: "llama3:latest" }),
      new Model({ model: "mistral:latest" }),
    ];

    const merged = mergeModels(localModels, true);

    expect(merged.filter((m) => m.isCloud()).length).toBe(0);

    for (let i = 0; i < FEATURED_MODELS.length; i++) {
      expect(merged[i].model).toBe(FEATURED_MODELS[i]);
    }

    const featuredCount = FEATURED_MODELS.length;
    expect(merged[featuredCount].model).toBe("llama3:latest");
    expect(merged[featuredCount + 1].model).toBe("mistral:latest");
  });

  it("should handle empty input", () => {
    const merged = mergeModels([]);

    expect(merged.every((m) => !m.isCloud())).toBe(true);
    for (let i = 0; i < FEATURED_MODELS.length; i++) {
      expect(merged[i].model).toBe(FEATURED_MODELS[i]);
    }
    expect(merged.length).toBe(FEATURED_MODELS.length);
  });

  it("should sort non-featured locals alphabetically before cloud entries", () => {
    const localModels: Model[] = [
      new Model({ model: "zephyr:latest" }),
      new Model({ model: "alpha:latest" }),
      new Model({ model: "gpt-oss:120b-cloud" }),
    ];

    const merged = mergeModels(localModels, false);

    expect(merged[merged.length - 1].model).toBe("gpt-oss:120b-cloud");

    for (let i = 0; i < FEATURED_MODELS.length; i++) {
      expect(merged[i].model).toBe(FEATURED_MODELS[i]);
    }

    const featuredCount = FEATURED_MODELS.length;
    expect(merged[featuredCount].model).toBe("alpha:latest");
    expect(merged[featuredCount + 1].model).toBe("zephyr:latest");
  });
});
