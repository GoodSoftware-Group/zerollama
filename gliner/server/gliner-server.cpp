// gliner-server — sibling HTTP for GLiNER.cpp (GL1).
// Dual wire: POST /v1/extract (abstract) + POST /v1/gliner (engine knobs).
// Lab ports only. Docs: docs/gliner-cpp.md

#include <GLiNER/gliner_config.hpp>
#include <GLiNER/model.hpp>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct ServerState {
  std::mutex mu;
  std::string model_path;
  std::string tokenizer_path;
  gliner::Config config{12, 512, gliner::SPAN_LEVEL};
  int device_id = -1; // -1 = CPU
  std::unique_ptr<gliner::Model> model;
};

void load_model(ServerState &st) {
  if (st.device_id >= 0) {
    st.model = std::make_unique<gliner::Model>(st.model_path, st.tokenizer_path, st.config, st.device_id);
  } else {
    st.model = std::make_unique<gliner::Model>(st.model_path, st.tokenizer_path, st.config);
  }
}

json spans_to_json(const std::vector<gliner::Span> &spans) {
  json arr = json::array();
  for (const auto &s : spans) {
    arr.push_back({
        {"text", s.text},
        {"label", s.classLabel},
        {"start", s.startIdx},
        {"end", s.endIdx},
        {"score", s.prob},
    });
  }
  return arr;
}

json engine_echo(const ServerState &st) {
  return {
      {"max_width", st.config.maxWidth},
      {"max_length", st.config.maxLength},
      {"model_type", st.config.modelType == gliner::TOKEN_LEVEL ? "token" : "span"},
      {"device_id", st.device_id},
  };
}

// Allowed top-level keys for abstract /v1/extract (extra keys → 400).
const std::vector<std::string> kExtractKeys = {
    "model", "text", "texts", "labels", "threshold",
};

// Allowed top-level keys for /v1/gliner (engine surface).
const std::vector<std::string> kGlinerKeys = {
    "model", "text", "texts", "labels", "threshold",
    "flat_ner", "multi_label",
    "max_width", "max_length", "model_type", "device_id",
};

bool unknown_keys(const json &body, const std::vector<std::string> &allowed, std::string &bad) {
  for (auto it = body.begin(); it != body.end(); ++it) {
    bool ok = false;
    for (const auto &a : allowed) {
      if (it.key() == a) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      bad = it.key();
      return true;
    }
  }
  return false;
}

bool parse_texts_labels(const json &body, std::vector<std::string> &texts, std::vector<std::string> &labels, std::string &err) {
  if (body.contains("texts") && body["texts"].is_array()) {
    for (const auto &t : body["texts"]) {
      if (!t.is_string()) {
        err = "texts[] must be strings";
        return false;
      }
      texts.push_back(t.get<std::string>());
    }
  } else if (body.contains("text") && body["text"].is_string()) {
    texts.push_back(body["text"].get<std::string>());
  } else {
    err = "text or texts required";
    return false;
  }
  if (!body.contains("labels") || !body["labels"].is_array() || body["labels"].empty()) {
    err = "labels[] required";
    return false;
  }
  for (const auto &l : body["labels"]) {
    if (!l.is_string()) {
      err = "labels[] must be strings";
      return false;
    }
    labels.push_back(l.get<std::string>());
  }
  return true;
}

void handle_infer(ServerState &st, const json &body, bool mechanical, httplib::Response &res) {
  res.set_header("content-type", "application/json");

  std::string bad;
  if (unknown_keys(body, mechanical ? kGlinerKeys : kExtractKeys, bad)) {
    res.status = 400;
    res.set_content(json{{"error", "unknown field: " + bad}}.dump(), "application/json");
    return;
  }

  std::vector<std::string> texts, labels;
  std::string err;
  if (!parse_texts_labels(body, texts, labels, err)) {
    res.status = 400;
    res.set_content(json{{"error", err}}.dump(), "application/json");
    return;
  }

  float threshold = 0.5f;
  if (body.contains("threshold") && body["threshold"].is_number()) {
    threshold = body["threshold"].get<float>();
  }
  bool flat_ner = true;
  bool multi_label = false;
  if (mechanical) {
    if (body.contains("flat_ner") && body["flat_ner"].is_boolean()) {
      flat_ner = body["flat_ner"].get<bool>();
    }
    if (body.contains("multi_label") && body["multi_label"].is_boolean()) {
      multi_label = body["multi_label"].get<bool>();
    }
  }

  std::lock_guard<std::mutex> lock(st.mu);
  if (!st.model) {
    res.status = 503;
    res.set_content(json{{"error", "model not loaded"}}.dump(), "application/json");
    return;
  }

  if (mechanical) {
    bool need_reload = false;
    gliner::Config want = st.config;
    int want_dev = st.device_id;
    if (body.contains("max_width") && body["max_width"].is_number_integer()) {
      int v = body["max_width"].get<int>();
      if (v != st.config.maxWidth) {
        want.maxWidth = v;
        need_reload = true;
      }
    }
    if (body.contains("max_length") && body["max_length"].is_number_integer()) {
      int v = body["max_length"].get<int>();
      if (v != st.config.maxLength) {
        want.maxLength = v;
        need_reload = true;
      }
    }
    if (body.contains("model_type") && body["model_type"].is_string()) {
      auto mt = body["model_type"].get<std::string>();
      gliner::ModelType t = (mt == "token") ? gliner::TOKEN_LEVEL : gliner::SPAN_LEVEL;
      if (mt != "token" && mt != "span") {
        res.status = 400;
        res.set_content(json{{"error", "model_type must be span|token"}}.dump(), "application/json");
        return;
      }
      if (t != st.config.modelType) {
        want.modelType = t;
        need_reload = true;
      }
    }
    if (body.contains("device_id") && body["device_id"].is_number_integer()) {
      int v = body["device_id"].get<int>();
      if (v != st.device_id) {
        want_dev = v;
        need_reload = true;
      }
    }
    if (need_reload) {
      try {
        st.config = want;
        st.device_id = want_dev;
        load_model(st);
      } catch (const std::exception &e) {
        res.status = 500;
        res.set_content(json{{"error", std::string("reload failed: ") + e.what()}}.dump(), "application/json");
        return;
      }
    }
  }

  std::vector<std::vector<gliner::Span>> out;
  try {
    out = st.model->inference(texts, labels, flat_ner, threshold, multi_label);
  } catch (const std::exception &e) {
    res.status = 500;
    res.set_content(json{{"error", std::string("inference failed: ") + e.what()}}.dump(), "application/json");
    return;
  }

  json resp;
  if (texts.size() == 1) {
    resp["entities"] = spans_to_json(out.empty() ? std::vector<gliner::Span>{} : out[0]);
  } else {
    json results = json::array();
    for (size_t i = 0; i < out.size(); ++i) {
      results.push_back({{"entities", spans_to_json(out[i])}});
    }
    resp["results"] = results;
  }
  if (mechanical) {
    resp["engine"] = engine_echo(st);
    resp["threshold"] = threshold;
    resp["flat_ner"] = flat_ner;
    resp["multi_label"] = multi_label;
  }
  res.status = 200;
  res.set_content(resp.dump(), "application/json");
}

void usage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --model MODEL.onnx --tokenizer tokenizer.json"
            << " [--host 127.0.0.1] [--port 18094]"
            << " [--max-width 12] [--max-length 512] [--model-type span|token]"
            << " [--device-id N]\n";
}

} // namespace

int main(int argc, char **argv) {
  ServerState st;
  std::string host = "127.0.0.1";
  int port = 18094;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") {
      st.model_path = need("--model");
    } else if (a == "--tokenizer") {
      st.tokenizer_path = need("--tokenizer");
    } else if (a == "--host") {
      host = need("--host");
    } else if (a == "--port") {
      port = std::stoi(need("--port"));
    } else if (a == "--max-width") {
      st.config.maxWidth = std::stoi(need("--max-width"));
    } else if (a == "--max-length") {
      st.config.maxLength = std::stoi(need("--max-length"));
    } else if (a == "--model-type") {
      auto mt = need("--model-type");
      st.config.modelType = (mt == "token") ? gliner::TOKEN_LEVEL : gliner::SPAN_LEVEL;
    } else if (a == "--device-id") {
      st.device_id = std::stoi(need("--device-id"));
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown arg: " << a << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (st.model_path.empty() || st.tokenizer_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  // Refuse production ports.
  if (port == 11434 || port == 8081 || port == 8080) {
    std::cerr << "error: refusing production port " << port << " (use lab e.g. 18094)\n";
    return 2;
  }

  try {
    load_model(st);
  } catch (const std::exception &e) {
    std::cerr << "error: load model: " << e.what() << "\n";
    return 1;
  }

  httplib::Server svr;
  svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
    res.set_content(R"({"ok":true})", "application/json");
  });
  svr.Get("/props", [&](const httplib::Request &, httplib::Response &res) {
    std::lock_guard<std::mutex> lock(st.mu);
    res.set_content(engine_echo(st).dump(), "application/json");
  });
  svr.Post("/v1/extract", [&](const httplib::Request &req, httplib::Response &res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"error":"invalid json"})", "application/json");
      return;
    }
    handle_infer(st, body, false, res);
  });
  svr.Post("/v1/gliner", [&](const httplib::Request &req, httplib::Response &res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"error":"invalid json"})", "application/json");
      return;
    }
    handle_infer(st, body, true, res);
  });

  std::cerr << "gliner-server listening http://" << host << ":" << port
            << " model=" << st.model_path << "\n";
  if (!svr.listen(host.c_str(), port)) {
    std::cerr << "error: listen failed\n";
    return 1;
  }
  return 0;
}
