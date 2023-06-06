#include <string>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>

#include <iostream>
#include <memory>
#include <string>
#include <queue>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "absl/log/initialize.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "proto/llama.grpc.pb.h"

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

uint64_t current_time_millis() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

class LlamaManager {
    private:
        const int number_of_threads;
        const int context_size_tokens;

        llama_context *ctx;

        std::vector<llama_token> computed_context;
        std::vector<llama_token> pending_context;

    public:
        LlamaManager() :
              number_of_threads (8)
            , context_size_tokens (2048)
            , ctx (nullptr)
        {
        }

        ~LlamaManager() {
            if (ctx != nullptr) {
                llama_free(ctx);
            }
        }

        llama_context* get_context() {
            return ctx;
        }

        std::vector<llama_token> get_computed_context() {
            return computed_context;
        }

        void load_model(const std::string& model_filename) {
            if (ctx) {
                throw std::runtime_error("Model already loaded");
            }

            llama_context_params lparams = llama_context_default_params();
            lparams.n_ctx = context_size_tokens;

            ctx = llama_init_from_file(model_filename.c_str(), lparams);

            if (ctx == nullptr) {
                throw std::runtime_error("Error: failed to initialize llama context");
            }

            computed_context.clear();
            pending_context.clear();
            pending_context.push_back(llama_token_bos());
        }

        void add_token(llama_token tok) {
            pending_context.push_back(tok);
        }

        void compute_logits() {
            const int n_batch_size = 8;

            while (pending_context.size() > 0) {
                const int n_tokens = std::min((int)pending_context.size(), n_batch_size);

                llama_token *tokens = pending_context.data();

                if (llama_eval(ctx, tokens, n_tokens, computed_context.size(), number_of_threads) != 0) {
                    throw std::runtime_error("Failed to evaluate tokens");
                }

                for (int i = 0; i < n_tokens; i++) {
                    computed_context.push_back(pending_context[i]);
                }
                pending_context.erase(pending_context.begin(), pending_context.begin() + n_tokens);
            }
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

ABSL_FLAG(uint16_t, port, 50051, "Server port for the service");

class LlamaServiceImpl final : public LlamaService::Service {
private:
    absl::Mutex mutex;
    std::unique_ptr<LlamaManager> llama_manager;

public:
  std::string map_model_filename(const std::string& model_name) {
    if (model_name == "13B") {
        return "/home/svk/dalai/llama/models/13B/ggml-model-q4_0.bin";
    }

    throw std::runtime_error("Unknown model: " + model_name);
  }

  Status DoLoadModel(ServerContext* context, const ::llamagrpc::DoLoadModelRequest* request, ::llamagrpc::DoLoadModelResponse* reply) override {
    absl::MutexLock lock(&mutex);

    const std::string mapped_model_filename = map_model_filename(request->model_name());

    LOG(INFO) << "Loading requested model: " << mapped_model_filename;

    llama_manager.reset(new LlamaManager());
    llama_manager->load_model(mapped_model_filename);

    LOG(INFO) << "Done loading requested model: " << mapped_model_filename;

    reply->set_model_ready(true);
    return Status::OK;
  }

  Status Tokenize(ServerContext* context, const ::llamagrpc::TokenizeRequest* request, ::llamagrpc::TokenizeResponse* reply) override {
    absl::MutexLock lock(&mutex);

    if (!llama_manager) {
        return Status(StatusCode::FAILED_PRECONDITION, "Model not loaded");
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

    if (!llama_manager) {
        return Status(StatusCode::FAILED_PRECONDITION, "Model not loaded");
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

  Status DoAddTokensAndCompute(ServerContext* context, const ::llamagrpc::DoAddTokensAndComputeRequest* request, ::llamagrpc::DoAddTokensAndComputeResponse* reply) override {
    absl::MutexLock lock(&mutex);

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

    return Status::OK;
  }

  Status DoSaveCheckpoint(ServerContext* context, const ::llamagrpc::DoSaveCheckpointRequest* request, ::llamagrpc::DoSaveCheckpointResponse* reply) override {
    absl::MutexLock lock(&mutex);

    if (!llama_manager) {
        return Status(StatusCode::FAILED_PRECONDITION, "Model not loaded");
    }

    std::string filename = "/tmp/llamagrpc.saved-checkpoint";

    llama_manager->save_checkpoint(filename);

    return Status::OK;
  }

  Status DoRestoreCheckpoint(ServerContext* context, const ::llamagrpc::DoRestoreCheckpointRequest* request, ::llamagrpc::DoRestoreCheckpointResponse* reply) override {
    absl::MutexLock lock(&mutex);

    if (!llama_manager) {
        return Status(StatusCode::FAILED_PRECONDITION, "Model not loaded");
    }

    std::string filename = "/tmp/llamagrpc.saved-checkpoint";

    llama_manager->restore_checkpoint(filename);

    return Status::OK;
  }
};

void RunServer(uint16_t port) {
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
  absl::ParseCommandLine(argc, argv);

  LOG(INFO) << "Starting up.";
  RunServer(absl::GetFlag(FLAGS_port));
  return 0;
}

#if 0
int main(int argc, char ** argv) {
    const int n_threads = 8;
    
    llama_init_backend();

    llama_context_params lparams = llama_context_default_params();
    lparams.n_ctx = 2048;

    const char * model_filename = "/home/svk/dalai/llama/models/13B/ggml-model-q4_0.bin";
    llama_context *ctx = llama_init_from_file(model_filename, lparams);

    if (ctx == nullptr) {
        fprintf(stderr, "Error: failed to initialize llama context\n");
        return 1;
    }

    std::string saved_session_filename = "/tmp/llama-saved-session";

    std::string prompt_text = read_text_file("/tmp/llama_prompt.txt");

    std::vector<llama_token> session_tokens;

    if (!file_exists(saved_session_filename)) {
        std::vector<llama_token> tokens = simple_tokenize(ctx, prompt_text);

        tokens.insert(tokens.begin(), llama_token_bos());

        auto t00 = current_time_millis();

        // Feed prompt into the model
        for (llama_token token: tokens) {
            printf("%d -- %s\n", token, llama_token_to_str(ctx, token));


            llama_token* tokens_to_feed = &token;
            const int n_tokens_to_feed = 1;
            const int n_past = session_tokens.size();

            if (session_tokens.size() >= 2048) {
                fprintf(stderr, "total fed %d; breaking off\n", session_tokens.size());
                break;
            }

            fprintf(stderr, "total fed %d\n", session_tokens.size());
            if (llama_eval(ctx, tokens_to_feed, n_tokens_to_feed, n_past, n_threads) != 0) {
                fprintf(stderr, "Error: failed to evaluate tokens\n");
                return 1;
            }

            session_tokens.push_back(token);
        }

        auto t0 = current_time_millis();

        fprintf(stderr, "loaded prompt in %d ms\n", t0 - t00);

        if (!llama_save_session_file(ctx, saved_session_filename.c_str(), session_tokens.data(), session_tokens.size())) {
            fprintf(stderr, "Error: failed to save session file\n");
            return 1;
        }

        auto t1 = current_time_millis();
        fprintf(stderr, "saved session in %d ms\n", t1 - t0);
    } else {
        session_tokens.resize(2048);
        size_t n_tokens;

        auto t0 = current_time_millis();
        if (!llama_load_session_file(ctx, saved_session_filename.c_str(), session_tokens.data(), session_tokens.size(), &n_tokens)) {
            fprintf(stderr, "Error: failed to load session file\n");
            return 1;
        }

        session_tokens.resize(n_tokens);

        auto t1 = current_time_millis();
        fprintf(stderr, "loaded session in %d ms\n", t1 - t0);
    }

    while (session_tokens.size() < 2048) {
        // Choose next token
        auto logits  = llama_get_logits(ctx);
        const int n_vocab = llama_n_vocab(ctx);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        float max_logit = -1e9;
        llama_token best_token = 0;
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            const float logit = logits[token_id];
            if (logit > max_logit) {
                max_logit = logit;
                best_token = token_id;
            }
        }

        // TODO: repetition penalties
        // Modify logit.

        // Temperature choice strategy:
        //   - Choose top K (40)
        //   - Apply softmax with configured temperature (0.8)

        llama_token chosen_token = best_token;
        if (llama_eval(ctx, &chosen_token, 1, session_tokens.size(), n_threads) != 0) {
            fprintf(stderr, "Error: failed to evaluate tokens\n");
            return 1;
        }
        session_tokens.push_back(chosen_token);

        const char * token_str = llama_token_to_str(ctx, best_token);
        fprintf(stdout, "%s", token_str);
        fflush(stdout);
    }

    /*
    API:

    All calls are mutexed.

    Actions:
        DoLoadModel(name)

        DoContextAppendAndCompute(token_id)
        DoContextSetAndCompute([tokens])

        DoRestoreCheckpoint
        DoSaveCheckpoint // affects file system only, but not model.

    State inspection:
        GetNextTokenLogits() // top N, or all.
        GetCurrentContext() -> [str, [tokens]]

    Utility functions:
        Tokenize(str) -> [(str, token_id)]
        GetTokens() -> [(str, token_id)]

        GetTopLogits(n) -> [(token_id, logit)]

    */

    return 0;
}

int main(int argc, char **argv) {
    llama_init_backend();

    LlamaManager llama;

    return 0;
}

#endif
