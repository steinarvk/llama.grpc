#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/log/check.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

#include "server/storage.h"

#include <arpa/inet.h>

#include "SHA256.h"

#define RETURN_IF_ERROR(expr)            \
  do {                                   \
    const absl::Status _status = (expr); \
    if (!_status.ok()) return _status;   \
  } while (0)


ABSL_FLAG(std::string, database_file, ":memory:", "Database filename");
ABSL_FLAG(std::string, checkpoint_dir, "/tmp/", "Checkpoint directory");

namespace llamagrpc {

int Storage::CURRENT_SCHEMA_VERSION = 1;

Storage::Storage() :
    database_filename(""),
    checkpoint_path(),
    db(nullptr)
{
}

bool Storage::Wrap(int rc) {
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "SQLite3 error: " << sqlite3_errmsg(db);
        return false;
    }
    return true;
}

void Storage::ExecOrDie(const char *sql) {
    char *errmsg;

    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errmsg);

    if (rc != SQLITE_OK) {
        LOG(FATAL) << "Fatal sqlite error: " << sqlite3_errmsg(db) << "; error: " << (errmsg ? errmsg : "") << "; sql: " << sql;
    }
    
    if (errmsg != nullptr) {
        sqlite3_free(errmsg);
    }
}

int Storage::GetSchemaVersionOrDie() {
    const char *sql = "SELECT schema_version FROM schema_version WHERE row_id = 1";
    sqlite3_stmt *stmt;

    CHECK(Wrap(sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr))) << "Failed to prepare statement: " << sql;

    int rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW || rc == SQLITE_DONE) << "Failed to execute statement: " << sql;
    CHECK(rc == SQLITE_ROW) << "No rows returned from statement: " << sql;

    int schema_version = sqlite3_column_int(stmt, 0);

    CHECK(Wrap(sqlite3_finalize(stmt)));

    return schema_version;
}

bool Storage::SchemaUpgradeIfNecessaryOrDie() {
    int schema_version = GetSchemaVersionOrDie();

    if (schema_version >= CURRENT_SCHEMA_VERSION) {
        LOG(INFO) << "Database ready at schema version " << schema_version << "; current schema version is " << CURRENT_SCHEMA_VERSION;
        return false;
    }

    switch (schema_version) {
        case 0:
            ExecOrDie(R"(
                BEGIN TRANSACTION;

                CREATE TABLE models (
                    model_id INTEGER PRIMARY KEY,
                    model_relative_path TEXT NOT NULL UNIQUE,
                    model_vocabulary_proto BLOB NULL
                );

                CREATE TABLE snapshots (
                    snapshot_id INTEGER PRIMARY KEY,
                    model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
                    timestamp_unix_seconds INTEGER NOT NULL,
                    snapshot_relative_path TEXT NOT NULL UNIQUE,
                    tokens_in_snapshot INTEGER NOT NULL,
                    file_size_bytes INTEGER NOT NULL
                );

                CREATE TABLE snapshotted_states (
                    snapshotted_state_id INTEGER PRIMARY KEY,
                    snapshot_id INTEGER NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
                    state_hash TEXT NOT NULL,
                    tokens_in_state INTEGER NOT NULL,
                    last_token_id INTEGER NOT NULL
                );

                UPDATE schema_version SET schema_version = 1;

                COMMIT;
            )");
            break;

        default:
            LOG(FATAL) << "Unknown schema version " << schema_version;
    }

    int new_schema_version = GetSchemaVersionOrDie();
    CHECK(new_schema_version > schema_version) << "Schema version did not increase after upgrade";

    LOG(INFO) << "Database upgraded from schema version " << schema_version << " to " << new_schema_version;

    return true;
}

void Storage::InitSchemaOrDie() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS schema_version (
            row_id INTEGER NOT NULL DEFAULT 1 CHECK (row_id = 1),
            schema_version INTEGER NOT NULL
        );

        INSERT OR IGNORE INTO schema_version (schema_version) VALUES (0);
    )";
    ExecOrDie(sql);
    while (SchemaUpgradeIfNecessaryOrDie());
}

void Storage::OpenDatabaseOrDie() {
    LOG(INFO) << "Opening database " << database_filename;
    CHECK(Wrap(sqlite3_open(database_filename.c_str(), &db)));

    ExecOrDie("PRAGMA foreign_keys = ON;");
}

Storage::~Storage() {
    if (db != nullptr) {
        LOG(INFO) << "Closing database " << database_filename;
        CHECK(Wrap(sqlite3_close(db))) << "Failed to close database " << database_filename;
    }
}

Storage* Storage::CreateFromFlags() {
    std::unique_ptr<Storage> storage(new Storage());

    storage->database_filename = absl::GetFlag(FLAGS_database_file);
    storage->checkpoint_path = absl::GetFlag(FLAGS_checkpoint_dir);
    storage->OpenDatabaseOrDie();
    storage->InitSchemaOrDie();

    return storage.release();
}

absl::StatusOr<uint64_t> GetFileSize(const std::filesystem::path& filename) {
    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        return absl::InternalError(absl::StrCat("Failed to stat file ", filename.c_str(), ": ", strerror(errno)));
    }
    return st.st_size;
}

absl::Status HashTokenSequence(llamagrpc_internal::TokenHashSequence* out, const llamagrpc_internal::TokenSequence& tokens) {
    if (tokens.token_id_size() == 0) {
        return absl::InvalidArgumentError("empty token sequence");
    }

    if (out->hexdigest_size() == tokens.token_id_size()) {
        return absl::OkStatus();
    }

    if (out->hexdigest_size() != 0) {
        return absl::InvalidArgumentError(absl::StrCat("TokenHashSequence already has ", out->hexdigest_size(), " elements; expected zero or ", tokens.token_id_size()));
    }

    SHA256 sha256;

    for (int i = 0; i < tokens.token_id_size(); i++) {
        uint32_t token_id = tokens.token_id(i);
        uint32_t network_byte_order_token_id = htonl(token_id);
        uint8_t *p = (uint8_t*) &network_byte_order_token_id;
        sha256.update(p, sizeof token_id);
        
        uint8_t *allocated_digest = sha256.digest();
        out->add_hexdigest(SHA256::toString(allocated_digest));
        delete [] allocated_digest;
    }

    return absl::OkStatus();
}

absl::Status Storage::RegisterSnapshot(const llamagrpc_internal::SnapshotDesc& desc) {
    LOG(INFO) << "Registering snapshot " << desc.DebugString();

    std::filesystem::path checkpoint_full_path = checkpoint_path / desc.snapshot_relative_path();
    absl::StatusOr<uint64_t> maybe_file_size_bytes = GetFileSize(checkpoint_full_path);
    if (!maybe_file_size_bytes.ok()) {
        return maybe_file_size_bytes.status();
    }

    uint64_t file_size_bytes = *maybe_file_size_bytes;
    
    LOG(INFO) << "Snapshot " << checkpoint_full_path << " exists and has " << file_size_bytes << " bytes";

    llamagrpc_internal::TokenHashSequence token_hash_sequence;
    RETURN_IF_ERROR(HashTokenSequence(&token_hash_sequence, desc.tokens()));
    
    const int n = desc.tokens().token_id_size();
    CHECK(n == token_hash_sequence.hexdigest_size());

    for (int i = 0; i < n; i++) {
        LOG(INFO) << "SHA256 after " << (i+1) << " tokens: " << token_hash_sequence.hexdigest(i) << " (token_id = " << desc.tokens().token_id(i) << ")";
    }

    return absl::OkStatus();
}

}