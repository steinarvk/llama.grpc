#include <string>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <unordered_map>

#include <iostream>
#include <memory>
#include <string>
#include <queue>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/log/check.h"
#include "absl/time/time.h"
#include "absl/log/initialize.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/random/random.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "proto/llama.grpc.pb.h"
#include "proto/internal/llamagrpc_internal.pb.h"

#include "server/storage.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

using ::llamagrpc::LlamaService;
using ::llamagrpc::DoLoadModelRequest;
using ::llamagrpc::DoLoadModelResponse;
using ::llamagrpc::TokenizeRequest;
using ::llamagrpc::TokenizeResponse;

#include "llama.h"

ABSL_FLAG(uint16_t, port, 50051, "Server port for the service");
ABSL_FLAG(int, threads, 0, "Number of threads");
ABSL_FLAG(int, batch_size, 512, "Batch size");
ABSL_FLAG(int, context_size, 2048, "Max context size");
ABSL_FLAG(std::string, model_dir, "", "Model directory");

struct ModelInfo {
    std::string model_name;
    std::filesystem::path model_path;
};

std::vector<llama_token> simple_tokenize(llama_context *ctx, std::string text) {
    const int n_max_tokens = text.size();

    std::vector<llama_token> tokens;
    tokens.resize(n_max_tokens);

    const int n_tokens = llama_tokenize(ctx, text.c_str(), tokens.data(), n_max_tokens, false);
    tokens.resize(n_tokens);

    return tokens;
}

bool file_exists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

std::unordered_map<std::string, ModelInfo> find_models() {
  std::unordered_map<std::string, ModelInfo> models;

  std::string model_dir_flag = absl::GetFlag(FLAGS_model_dir);
  
  if (model_dir_flag.empty()) {
    LOG(FATAL) << "--model_dir must be specified";
  }

  std::filesystem::path model_dir(model_dir_flag);
  model_dir = std::filesystem::canonical(model_dir);

  std::string prefix_path = model_dir.string() + "/";
  for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(model_dir)) {
    if ( entry.is_regular_file() ) {
        const auto path = entry.path();
        if (path.extension() == ".bin") {
            std::string model_full_path = path.string();
            std::string model_extension = path.extension();
            std::string model_name = std::string(absl::StripSuffix(absl::StripPrefix(model_full_path, prefix_path), model_extension));

            if (model_name == "ggml-vocab") {
                continue;
            }

            if (models.find(model_name) != models.end()) {
                LOG(FATAL) << "Found multiple models with the same name: " << model_name;
            }

            LOG(INFO) << "Found model: " << model_name << " " << model_full_path << " " << std::endl;
            models[model_name] = ModelInfo{model_name, path};
        }
    }
  }

  return models;
}

std::string read_text_file(const std::string& filename)
{
    std::ifstream inFile(filename);
    
    // check if file is open
    if (!inFile) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << inFile.rdbuf();
    return buffer.str();
}

std::string generate_session_id() {
    static const char *alphabet = "abcdef0123456789";
    const int sz = strlen(alphabet);
    const int n_chars = 32;
    std::string rv = "";

    absl::BitGen gen;

    for (int i = 0; i < n_chars; i++) {
        rv += alphabet[absl::Uniform(gen, 0, sz)];
    }

    return rv;
}

uint64_t current_time_millis() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int get_number_of_threads() {
    int n_threads = absl::GetFlag(FLAGS_threads);
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        LOG(INFO) << "Autodetected number of threads: " << n_threads << " (use --threads to override)";
    }
    return n_threads;
}

class LlamaManager {
    private:
        const int number_of_threads;
        const int context_size_tokens;
        std::string session_id;

        ::llamagrpc::SessionInfo session_info;

        llama_context *ctx;

        std::vector<llama_token> computed_context;
        std::vector<llama_token> pending_context;

    public:
        LlamaManager(int n_threads, int n_context, std::string model_name, const std::filesystem::path& model_path) :
              number_of_threads (n_threads)
            , context_size_tokens (n_context)
            , session_id (generate_session_id())
            , ctx (nullptr)
        {
            LOG(INFO) << "Initializing session " << session_id;

            llama_context_params lparams = llama_context_default_params();
            lparams.n_ctx = context_size_tokens;

            ctx = llama_init_from_file(model_path.c_str(), lparams);

            if (ctx == nullptr) {
                throw std::runtime_error("Error: failed to initialize llama context");
            }

            pending_context.push_back(llama_token_bos());

            session_info.set_session_id(session_id);
            session_info.set_model_name(model_name);
            session_info.set_start_time_unix_nanos(absl::ToUnixNanos(absl::Now()));

            LOG(INFO) << "Initialized session " << session_id;
        }

        ~LlamaManager() {
            if (ctx != nullptr) {
                llama_free(ctx);
            }
        }

        const int get_context_size() {
            return context_size_tokens;
        }

        const ::llamagrpc::SessionInfo& get_session_info() {
            return session_info;
        }

        const std::string& get_session_id() {
            return session_id;
        }

        llama_context* get_context() {
            return ctx;
        }

        std::vector<llama_token> get_computed_context() {
            return computed_context;
        }

        bool is_model(const llamagrpc::ModelInfo& model_info) {
            return model_info.model_name() == session_info.model_name();
        }

        std::string regenerate_session_id() {
            session_id = generate_session_id();
            session_info.set_session_id(session_id);
            return session_info.session_id();
        }

        int get_common_prefix_size(const std::vector<llama_token>& other_tokens) {
            size_t i = 0;

            for (; i < computed_context.size() && i < other_tokens.size(); i++) {
                if (computed_context[i] != other_tokens[i]) {
                    break;
                }
            }

            return (int) i;
        }

        int get_remaining_context_size() {
            return context_size_tokens - (computed_context.size() + pending_context.size());
        }

        void truncate_computed_context(int n_tokens) {
            size_t sz = (size_t) n_tokens;
            // If we have more than n_tokens in the computed context, truncate it by erasing the excess tokens at the end
            if (computed_context.size() > sz) {
                computed_context.erase(computed_context.begin() + sz, computed_context.end());
                CHECK(computed_context.size() == sz);
            }
        }

        void clear_context() {
            computed_context.clear();
            pending_context.clear();
            pending_context.push_back(llama_token_bos());
        }

        void add_token(llama_token tok) {
            if (tok == llama_token_bos()) {
                const int total_current_queue = (int) (computed_context.size() + pending_context.size());
                bool do_ignore = false;

                if (total_current_queue == 1) {
                    if (computed_context.size() == 1 && computed_context[0] == llama_token_bos()) {
                        do_ignore = true;
                    }
                    if (pending_context.size() == 1 && pending_context[0] == llama_token_bos()) {
                        do_ignore = true;
                    }
                }

                if (do_ignore) {
                    LOG(INFO) << "Ignoring initial BOS token";
                    return;
                }
            }

            DLOG(INFO) << "Adding token: " << tok << "(string form:" << llama_token_to_str(ctx, tok) << ")";
            pending_context.push_back(tok);
        }

        void compute_logits() {
            const int n_batch_size = absl::GetFlag(FLAGS_batch_size);

            absl::Time t0 = absl::Now();

            if (pending_context.empty()) {
                llama_token *tokens = computed_context.data();
                tokens += computed_context.size() - 1;
                LOG(INFO) << "Re-evaluating token " << *tokens;
                if (llama_eval(ctx, tokens, 1, computed_context.size() - 1, number_of_threads) != 0) {
                    throw std::runtime_error("Failed to evaluate tokens");
                }

                absl::Time t1 = absl::Now();
                LOG(INFO) << "Re-evaluated one token in " << absl::ToDoubleMilliseconds(t1 - t0) << "ms";
                return;
            }

            const int pending_context_size = (int) pending_context.size();

            std::string all_pending_bytes;
            for (llama_token tok : pending_context) {
                std::string tok_bytes (llama_token_to_str(ctx, tok));
                all_pending_bytes.append(tok_bytes);
            }
            LOG(INFO) << "Evaluating pending context of " << pending_context_size << " tokens in batches of " << n_batch_size << " tokens (building on top of computed context of " << computed_context.size() << "), new bytes are: " << all_pending_bytes;

            while (pending_context.size() > 0) {
                const int n_tokens = std::min((int)pending_context.size(), n_batch_size);

                llama_token *tokens = pending_context.data();

                LOG(INFO) << "Evaluating " << n_tokens << " tokens after computed context of " << computed_context.size() << " tokens";
                const int show_max_seq = 5;
                const bool show_all = n_tokens <= (show_max_seq * 2);
                for (int i = 0; i < n_tokens; i++) {
                    bool should_show = show_all || (i < show_max_seq) || (i >= (n_tokens - show_max_seq));
                    if (should_show) {
                        LOG(INFO) << "token " << i << ": " << tokens[i] << "[" << llama_token_to_str(ctx, tokens[i]) << "]";
                    } else if (i == show_max_seq) {
                        LOG(INFO) << "...";
                    }
                }

                if (llama_eval(ctx, tokens, n_tokens, computed_context.size(), number_of_threads) != 0) {
                    throw std::runtime_error("Failed to evaluate tokens");
                }

                for (int i = 0; i < n_tokens; i++) {
                    computed_context.push_back(pending_context[i]);
                }
                pending_context.erase(pending_context.begin(), pending_context.begin() + n_tokens);
            }

            absl::Time t1 = absl::Now();
            double elapsed_ms = absl::ToDoubleMilliseconds(t1 - t0);
            LOG(INFO) << "Evaluated " << pending_context_size << " tokens in " << elapsed_ms << "ms; " << (1000.0 * ((double) pending_context_size) / elapsed_ms) << " tokens/sec";
        }

        void save_checkpoint(std::string filename) {
            if (!llama_save_session_file(ctx, filename.c_str(), computed_context.data(), computed_context.size())) {
                throw std::runtime_error("Failed to save checkpoint");
            }
        }

        void restore_checkpoint(std::string filename) {
            computed_context.clear();
            pending_context.clear();

            computed_context.resize(4096);

            size_t n_tokens;

            if (!llama_load_session_file(ctx, filename.c_str(), computed_context.data(), computed_context.size(), &n_tokens)) {
                throw std::runtime_error("Failed to load checkpoint");
            }

            computed_context.resize(n_tokens);
        }
};

class LlamaServiceImpl final : public LlamaService::Service {
private:
    const int n_threads;

    absl::Mutex mutex;
    std::unique_ptr<LlamaManager> llama_manager;
    std::unordered_map<std::string, ModelInfo> models;

public:
  LlamaServiceImpl() :
    n_threads ( get_number_of_threads() )
  {
    models = find_models();
  }

  std::filesystem::path map_model_filename(const std::string& model_name) {
    const auto& it = models.find(model_name);
    if (it == models.end()) {
        throw std::runtime_error("Unknown model: " + model_name);
    }

    return it->second.model_path;
  }

  Status DoLoadModel(ServerContext* context, const ::llamagrpc::DoLoadModelRequest* request, ::llamagrpc::DoLoadModelResponse* reply) override {
    absl::MutexLock lock(&mutex);

    const int n_context = absl::GetFlag(FLAGS_context_size);

    std::string model_name = request->model_name();
    std::string model_filename = map_model_filename(model_name);

    llama_manager.reset(new LlamaManager(n_threads, n_context, model_name, model_filename));

    reply->set_model_ready(true);
    reply->mutable_session_info()->CopyFrom(llama_manager->get_session_info());

    return Status::OK;
  }

  Status Tokenize(ServerContext* context, const ::llamagrpc::TokenizeRequest* request, ::llamagrpc::TokenizeResponse* reply) override {
    absl::MutexLock lock(&mutex);

    {
        auto status = WithLockPrepareModel(request->model_info(), request->session_hint(), false);
        if (!status.ok()) {
            return status;
        }
    }

    const std::string text = request->text();
    const std::vector<llama_token> tokens = simple_tokenize(llama_manager->get_context(), text);

    for (llama_token tok : tokens) {
        const char *token_str = llama_token_to_str(llama_manager->get_context(), tok);

        ::llamagrpc::Token* token_msg = reply->add_token();
        token_msg->set_token_id(tok);
        token_msg->set_token_str(token_str);
    }

    return Status::OK;
  }

  Status GetVocabulary(ServerContext* context, const ::llamagrpc::GetVocabularyRequest* request, ::llamagrpc::GetVocabularyResponse* reply) override {
    absl::MutexLock lock(&mutex);

    Status status = EnsureSessionLoaded("");
    if (!status.ok()) {
        return status;
    }

    const int n_vocab = llama_n_vocab(llama_manager->get_context());

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        const char *token_str = llama_token_to_str(llama_manager->get_context(), token_id);
        ::llamagrpc::Token* token_msg = reply->add_token();

        token_msg->set_token_id(token_id);
        token_msg->set_token_str(token_str);
    }

    return Status::OK;
  }

  Status EnsureSessionLoaded(const std::string session_id) {
    if (!llama_manager) {
        return Status(StatusCode::FAILED_PRECONDITION, "No model loaded");
    }

    // Anything will do if nothing is requested.
    if (!session_id.empty()) {
        if (llama_manager->get_session_id() != session_id) {
            return Status(StatusCode::FAILED_PRECONDITION, absl::StrCat("session_id requested:", session_id, " is not loaded"));
        }
    }

    return Status::OK;
  }

  grpc::Status WithLockPrepareModel(const ::llamagrpc::ModelInfo& model_info, const ::llamagrpc::SessionHint& session_hint, bool need_session = true) {
    std::string session_id = session_hint.session_id();

    if (model_info.model_name().empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "model_name is required");
    }

    // Do we have a session, meaning we don't need to reload the model?

    if (!session_id.empty() && EnsureSessionLoaded(session_id).ok()) {
        // Session is already loaded, so we can just use it.
    } else if (llama_manager && llama_manager->is_model(model_info)) {
        // TODO: as long as there's only 1 session active, we could actually save time here by not reloading the model
        //       if the new model requested is the same as the old model.
        std::string old_session_id = llama_manager->get_session_id();

        if (need_session) {
            session_id = llama_manager->regenerate_session_id();
            LOG(INFO) << "Reusing old session " << old_session_id << " as " << session_id;
        }
    } else {
        const int n_context = absl::GetFlag(FLAGS_context_size);

        std::string model_name = model_info.model_name();
        std::string model_filename = map_model_filename(model_name);

        LOG(INFO) << "Creating new session"; 

        llama_manager.reset(new LlamaManager(n_threads, n_context, model_name, model_filename));

        session_id = llama_manager->get_session_id();

        LOG(INFO) << "Initialized model for new session " << session_id;
    }

    return Status::OK;
  }

  Status DoPredict(ServerContext* context, const ::llamagrpc::DoPredictRequest* request, ::llamagrpc::DoPredictResponse* reply) override {
    absl::MutexLock lock(&mutex);

    {
        auto status = WithLockPrepareModel(request->model_info(), request->session_hint(), true);
        if (!status.ok()) {
            return status;
        }
    }

    // Now for the tokens.

    const int model_context_size = llama_manager->get_context_size();

    LOG(INFO) << "Tokenizing input; expecting at most " << model_context_size << " tokens";

    auto maybe_tokenized = ConvertInputTokens(request->full_context());
    if (!maybe_tokenized.ok()) {
        return Status(StatusCode::INVALID_ARGUMENT, "Failed to tokenize input");
    }
    std::vector<llama_token> tokenized = maybe_tokenized.value();
    const int actual_tokens = (int) tokenized.size();

    if (actual_tokens > model_context_size) {
        return Status(StatusCode::INVALID_ARGUMENT, "Too many tokens");
    }
    if (actual_tokens == 0) {
        return Status(StatusCode::INVALID_ARGUMENT, "No tokens");
    }

    LOG(INFO) << "Getting common prefix size with new " << tokenized.size() << " tokens";
    const int common_prefix_size = llama_manager->get_common_prefix_size(tokenized);

    CHECK(common_prefix_size <= (int) tokenized.size());

    LOG(INFO) << "Truncating context to " << common_prefix_size << " tokens";

    llama_manager->truncate_computed_context(common_prefix_size);

    // TODO: see if we can do better by loading a snapshot.

    const int n_input_tokens = (int) tokenized.size();

    if (common_prefix_size < n_input_tokens) {
        LOG(INFO) << "Adding more tokens from " << common_prefix_size << " to " << n_input_tokens;
        for (int i = common_prefix_size; i < n_input_tokens; i++) {
            llama_manager->add_token(tokenized[i]);
        }
    }

    LOG(INFO) << "Computing logits";

    llama_manager->compute_logits();

    const int n_vocab = llama_n_vocab(llama_manager->get_context());

    float* logits = llama_get_logits(llama_manager->get_context());

    LOG(INFO) << "Done computing logits";

    const ::llamagrpc::LogitProcessing& logit_processing = request->logit_processing();

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }                                                                   

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    if (logit_processing.has_llama_repetition_penalty()) {
        std::vector<llama_token> previous_tokens = llama_manager->get_computed_context();
        llama_context* ctx = llama_manager->get_context();

        float nl_logit = logits[llama_token_nl()];
        const float repeat_penalty = logit_processing.llama_repetition_penalty().intensity();
        llama_sample_repetition_penalty(ctx, &candidates_p, previous_tokens.data(), previous_tokens.size(), repeat_penalty);
        logits[llama_token_nl()] = nl_logit;
    } else {
        LOG(WARNING) << "No llama_repetition_penalty specified";
    }

    // TODO apply filters

    std::priority_queue<std::pair<float, llama_token>> top_tokens;

    for (llama_token_data token_data : candidates) {
        float logit = token_data.logit;
        llama_token token_id = token_data.id;

        top_tokens.push(std::make_pair(logit, token_id));
    }

    int n = logit_processing.top_n() == 0 ? n_vocab : logit_processing.top_n();

    LOG(INFO) << "Selecting top " << n << " tokens";

    for (int i = 0; i < n; i++) {
        std::pair<float, llama_token> top_token = top_tokens.top();

        const char *token_str = llama_token_to_str(llama_manager->get_context(), top_token.second);

        ::llamagrpc::TokenLogit* logit = reply->add_next_token_logit();
        logit->set_logit(top_token.first);
        logit->mutable_token()->set_token_id(top_token.second);
        logit->mutable_token()->set_token_str(token_str);

        top_tokens.pop();
    }

    reply->mutable_session_info()->CopyFrom(llama_manager->get_session_info());

    for (llama_token token_id : llama_manager->get_computed_context()) {
        uint32_t token_id_u32 = (uint32_t) token_id;
        reply->mutable_full_input_context()->add_token_id(token_id_u32);
    }

    return Status::OK;
  }

  absl::StatusOr<std::vector<llama_token>> ConvertInputTokens(const ::llamagrpc::InputTokens& input_tokens) {
    if (input_tokens.has_str()) {
        std::string untokenized_string = input_tokens.str();
        return simple_tokenize(llama_manager->get_context(), untokenized_string);
    }

    if (input_tokens.has_token_ids()) {
        std::vector<llama_token> tokenized;
        for (int i = 0; i < input_tokens.token_ids().token_id_size(); i++) {
            tokenized.push_back((llama_token) input_tokens.token_ids().token_id(i));
        }
        return tokenized;
    }

    return absl::Status(absl::StatusCode::kInvalidArgument, "No input tokens specified");
  }

  Status DoAddTokensAndCompute(ServerContext* context, const ::llamagrpc::DoAddTokensAndComputeRequest* request, ::llamagrpc::DoAddTokensAndComputeResponse* reply) override {
    absl::MutexLock lock(&mutex);

    Status status = EnsureSessionLoaded(request->session_id());
    if (!status.ok()) {
        return status;
    }
    reply->mutable_session_info()->CopyFrom(llama_manager->get_session_info());

    if (request->clear_context_first()) {
        llama_manager->clear_context();
    }

    std::string untokenized_string = request->input_tokens().str();
    std::vector<llama_token> tokenized = simple_tokenize(llama_manager->get_context(), untokenized_string);

    absl::Time t0 = absl::Now();

    LOG(INFO) << "Adding " << tokenized.size() << " tokens to the context";

    for (llama_token tok : tokenized) {
        llama_manager->add_token(tok);
    }

    LOG(INFO) << "Computing logits";

    llama_manager->compute_logits();

    const int n_vocab = llama_n_vocab(llama_manager->get_context());

    LOG(INFO) << "Done computing logits; picking top " << request->top_n_logits() << " tokens from the vocabulary of size " << n_vocab;

    std::priority_queue<std::pair<float, llama_token>> top_tokens;
    float* logits = llama_get_logits(llama_manager->get_context());

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }                                                                   

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    std::vector<llama_token> previous_tokens = llama_manager->get_computed_context();
    llama_context* ctx = llama_manager->get_context();

    float nl_logit = logits[llama_token_nl()];
    const float repeat_penalty = 1.1;
    llama_sample_repetition_penalty(ctx, &candidates_p, previous_tokens.data(), previous_tokens.size(), repeat_penalty);
    logits[llama_token_nl()] = nl_logit;


    for (llama_token_data token_data : candidates) {
        float logit = token_data.logit;
        llama_token token_id = token_data.id;

        top_tokens.push(std::make_pair(logit, token_id));
    }

    const int n = request->top_n_logits();

    for (int i = 0; i < n; i++) {
        std::pair<float, llama_token> top_token = top_tokens.top();

        const char *token_str = llama_token_to_str(llama_manager->get_context(), top_token.second);

        ::llamagrpc::TokenLogit* logit = reply->add_logit();
        logit->set_logit(top_token.first);
        logit->mutable_token()->set_token_id(top_token.second);
        logit->mutable_token()->set_token_str(token_str);

        top_tokens.pop();
    }


    LOG(INFO) << "Done picking top tokens";

    absl::Time t1 = absl::Now();
    absl::Duration d = t1 - t0;

    LOG(INFO) << "Added " << tokenized.size() << " tokens and computed logits in " << absl::ToDoubleSeconds(d) << " seconds";

    const int n_context_used = llama_manager->get_computed_context().size();
    reply->set_context_size_tokens(n_context_used);
    reply->set_remaining_context_size_tokens(llama_manager->get_remaining_context_size());

    return Status::OK;
  }

  Status DoSaveCheckpoint(ServerContext* context, const ::llamagrpc::DoSaveCheckpointRequest* request, ::llamagrpc::DoSaveCheckpointResponse* reply) override {
    absl::MutexLock lock(&mutex);

    Status status = EnsureSessionLoaded(request->session_id());
    if (!status.ok()) {
        return status;
    }
    reply->mutable_session_info()->CopyFrom(llama_manager->get_session_info());

    std::string filename = "/tmp/llamagrpc.saved-checkpoint";

    llama_manager->save_checkpoint(filename);

    return Status::OK;
  }

  Status DoRestoreCheckpoint(ServerContext* context, const ::llamagrpc::DoRestoreCheckpointRequest* request, ::llamagrpc::DoRestoreCheckpointResponse* reply) override {
    absl::MutexLock lock(&mutex);

    Status status = EnsureSessionLoaded(request->session_id());
    if (!status.ok()) {
        return status;
    }
    reply->mutable_session_info()->CopyFrom(llama_manager->get_session_info());

    std::string filename = "/tmp/llamagrpc.saved-checkpoint";

    llama_manager->restore_checkpoint(filename);

    return Status::OK;
  }
};

void RunServer(uint16_t port) {
  std::unique_ptr<llamagrpc::Storage> storage (llamagrpc::Storage::CreateFromFlags());

#if 0
  llamagrpc_internal::SnapshotDesc desc;
  desc.set_snapshot_relative_path("hello/mysnapshot.llamagrpc_snapshot");
  for (int i = 0; i < 3; i++) {
    desc.mutable_tokens()->add_token_id(i);
  }

  CHECK_OK(storage->RegisterSnapshot(desc));
#endif

  std::string server_address = absl::StrFormat("0.0.0.0:%d", port);
  LlamaServiceImpl service;

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  ServerBuilder builder;

  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());

  LOG(INFO) << "Server listening on " << server_address;

  server->Wait();
}

int main(int argc, char** argv) {
  LOG(INFO) << "Starting up.";

  absl::ParseCommandLine(argc, argv);
  
  RunServer(absl::GetFlag(FLAGS_port));
  return 0;
}